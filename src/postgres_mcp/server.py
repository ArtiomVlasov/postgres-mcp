# ruff: noqa: B008
import argparse
import asyncio
import logging
import os
import signal
import sys
from enum import Enum
from typing import Any
from typing import List
from typing import Literal
from typing import Union

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field
from pydantic import validate_call

from postgres_mcp.index.dta_calc import DatabaseTuningAdvisor

from .artifacts import ErrorResult
from .artifacts import ExplainPlanArtifact
from .database_health import DatabaseHealthTool
from .database_health import HealthType
from .explain import ExplainPlanTool
from .index.index_opt_base import MAX_NUM_INDEX_TUNING_QUERIES
from .index.llm_opt import LLMOptimizerTool
from .index.presentation import TextPresentation
from .sql import DatabaseRegistry
from .sql import SafeSqlDriver
from .sql import SqlDriver
from .sql import check_hypopg_installation_status
from .sql import obfuscate_password
from .sql.db_registry import DEFAULT_IDLE_TTL_SECONDS
from .sql.db_registry import DEFAULT_MAX_POOLS
from .top_queries import TopQueriesCalc

# Initialize FastMCP with default settings
mcp = FastMCP("postgres-mcp")

# Constants
PG_STAT_STATEMENTS = "pg_stat_statements"
HYPOPG_EXTENSION = "hypopg"
DATABASE_URI_HEADER = "x-database-uri"
ALIAS_ENV_PREFIX = "DATABASE_URI_"

DATABASE_PARAM_DESCRIPTION = (
    "Optional. Database to run against: an alias from list_databases, or a postgresql:// connection URI. Omit to use the server's default database."
)

ResponseType = List[types.TextContent | types.ImageContent | types.EmbeddedResource]

logger = logging.getLogger(__name__)


class AccessMode(str, Enum):
    """SQL access modes for the server."""

    UNRESTRICTED = "unrestricted"  # Unrestricted access
    RESTRICTED = "restricted"  # Read-only with safety features


# Global variables
db_registry = DatabaseRegistry()
current_access_mode = AccessMode.UNRESTRICTED
shutdown_in_progress = False


def _database_from_request_headers() -> str | None:
    """Value of the X-Database-URI header for HTTP transports, None for stdio or outside a request."""
    try:
        request = mcp.get_context().request_context.request
    except (LookupError, ValueError):
        return None
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    return headers.get(DATABASE_URI_HEADER)


async def get_sql_driver(database: str | None = None) -> Union[SqlDriver, SafeSqlDriver]:
    """Get the SQL driver for the requested database, based on the current access mode.

    ``database`` is an alias or connection URI; when omitted, the X-Database-URI header
    and then the server's default database are used.
    """
    if not isinstance(database, str):  # tools called directly may pass pydantic FieldInfo defaults
        database = None
    uri = db_registry.resolve(database, _database_from_request_headers())
    conn = await db_registry.get_pool(uri)
    base_driver = SqlDriver(conn=conn)

    if current_access_mode == AccessMode.RESTRICTED:
        logger.debug("Using SafeSqlDriver with restrictions (RESTRICTED mode)")
        return SafeSqlDriver(sql_driver=base_driver, timeout=30)  # 30 second timeout
    else:
        logger.debug("Using unrestricted SqlDriver (UNRESTRICTED mode)")
        return base_driver


def format_text_response(text: Any) -> ResponseType:
    """Format a text response."""
    return [types.TextContent(type="text", text=str(text))]


def format_error_response(error: str) -> ResponseType:
    """Format an error response."""
    return format_text_response(f"Error: {error}")


@mcp.tool(
    description="List the databases this server can work with (aliases usable as the 'database' argument of other tools) "
    "and whether arbitrary connection URIs are accepted.",
    annotations=ToolAnnotations(
        title="List Databases",
        readOnlyHint=True,
    ),
)
async def list_databases() -> ResponseType:
    """List configured databases without exposing credentials."""
    return format_text_response(db_registry.describe())


@mcp.tool(
    description="List all schemas in the database",
    annotations=ToolAnnotations(
        title="List Schemas",
        readOnlyHint=True,
    ),
)
async def list_schemas(
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """List all schemas in the database."""
    try:
        sql_driver = await get_sql_driver(database)
        rows = await sql_driver.execute_query(
            """
            SELECT
                schema_name,
                schema_owner,
                CASE
                    WHEN schema_name LIKE 'pg_%' THEN 'System Schema'
                    WHEN schema_name = 'information_schema' THEN 'System Information Schema'
                    ELSE 'User Schema'
                END as schema_type
            FROM information_schema.schemata
            ORDER BY schema_type, schema_name
            """
        )
        schemas = [row.cells for row in rows] if rows else []
        return format_text_response(schemas)
    except Exception as e:
        logger.error(f"Error listing schemas: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="List objects in a schema",
    annotations=ToolAnnotations(
        title="List Objects",
        readOnlyHint=True,
    ),
)
async def list_objects(
    schema_name: str = Field(description="Schema name"),
    object_type: str = Field(description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """List objects of a given type in a schema."""
    try:
        sql_driver = await get_sql_driver(database)

        if object_type in ("table", "view"):
            table_type = "BASE TABLE" if object_type == "table" else "VIEW"
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = {} AND table_type = {}
                ORDER BY table_name
                """,
                [schema_name, table_type],
            )
            objects = (
                [{"schema": row.cells["table_schema"], "name": row.cells["table_name"], "type": row.cells["table_type"]} for row in rows]
                if rows
                else []
            )

        elif object_type == "sequence":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT sequence_schema, sequence_name, data_type
                FROM information_schema.sequences
                WHERE sequence_schema = {}
                ORDER BY sequence_name
                """,
                [schema_name],
            )
            objects = (
                [{"schema": row.cells["sequence_schema"], "name": row.cells["sequence_name"], "data_type": row.cells["data_type"]} for row in rows]
                if rows
                else []
            )

        elif object_type == "extension":
            # Extensions are not schema-specific
            rows = await sql_driver.execute_query(
                """
                SELECT extname, extversion, extrelocatable
                FROM pg_extension
                ORDER BY extname
                """
            )
            objects = (
                [{"name": row.cells["extname"], "version": row.cells["extversion"], "relocatable": row.cells["extrelocatable"]} for row in rows]
                if rows
                else []
            )

        else:
            return format_error_response(f"Unsupported object type: {object_type}")

        return format_text_response(objects)
    except Exception as e:
        logger.error(f"Error listing objects: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Show detailed information about a database object",
    annotations=ToolAnnotations(
        title="Get Object Details",
        readOnlyHint=True,
    ),
)
async def get_object_details(
    schema_name: str = Field(description="Schema name"),
    object_name: str = Field(description="Object name"),
    object_type: str = Field(description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """Get detailed information about a database object."""
    try:
        sql_driver = await get_sql_driver(database)

        if object_type in ("table", "view"):
            # Get columns
            col_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = {} AND table_name = {}
                ORDER BY ordinal_position
                """,
                [schema_name, object_name],
            )
            columns = (
                [
                    {
                        "column": r.cells["column_name"],
                        "data_type": r.cells["data_type"],
                        "is_nullable": r.cells["is_nullable"],
                        "default": r.cells["column_default"],
                    }
                    for r in col_rows
                ]
                if col_rows
                else []
            )

            # Get constraints
            con_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT tc.constraint_name, tc.constraint_type, kcu.column_name
                FROM information_schema.table_constraints AS tc
                LEFT JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = {} AND tc.table_name = {}
                """,
                [schema_name, object_name],
            )

            constraints = {}
            if con_rows:
                for row in con_rows:
                    cname = row.cells["constraint_name"]
                    ctype = row.cells["constraint_type"]
                    col = row.cells["column_name"]

                    if cname not in constraints:
                        constraints[cname] = {"type": ctype, "columns": []}
                    if col:
                        constraints[cname]["columns"].append(col)

            constraints_list = [{"name": name, **data} for name, data in constraints.items()]

            # Get indexes
            idx_rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = {} AND tablename = {}
                """,
                [schema_name, object_name],
            )

            indexes = [{"name": r.cells["indexname"], "definition": r.cells["indexdef"]} for r in idx_rows] if idx_rows else []

            result = {
                "basic": {"schema": schema_name, "name": object_name, "type": object_type},
                "columns": columns,
                "constraints": constraints_list,
                "indexes": indexes,
            }

        elif object_type == "sequence":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT sequence_schema, sequence_name, data_type, start_value, increment
                FROM information_schema.sequences
                WHERE sequence_schema = {} AND sequence_name = {}
                """,
                [schema_name, object_name],
            )

            if rows and rows[0]:
                row = rows[0]
                result = {
                    "schema": row.cells["sequence_schema"],
                    "name": row.cells["sequence_name"],
                    "data_type": row.cells["data_type"],
                    "start_value": row.cells["start_value"],
                    "increment": row.cells["increment"],
                }
            else:
                result = {}

        elif object_type == "extension":
            rows = await SafeSqlDriver.execute_param_query(
                sql_driver,
                """
                SELECT extname, extversion, extrelocatable
                FROM pg_extension
                WHERE extname = {}
                """,
                [object_name],
            )

            if rows and rows[0]:
                row = rows[0]
                result = {"name": row.cells["extname"], "version": row.cells["extversion"], "relocatable": row.cells["extrelocatable"]}
            else:
                result = {}

        else:
            return format_error_response(f"Unsupported object type: {object_type}")

        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error getting object details: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Explains the execution plan for a SQL query, showing how the database will execute it and provides detailed cost estimates.",
    annotations=ToolAnnotations(
        title="Explain Query",
        readOnlyHint=True,
    ),
)
async def explain_query(
    sql: str = Field(description="SQL query to explain"),
    analyze: bool = Field(
        description="When True, actually runs the query to show real execution statistics instead of estimates. "
        "Takes longer but provides more accurate information.",
        default=False,
    ),
    hypothetical_indexes: list[dict[str, Any]] = Field(
        description="""A list of hypothetical indexes to simulate. Each index must be a dictionary with these keys:
    - 'table': The table name to add the index to (e.g., 'users')
    - 'columns': List of column names to include in the index (e.g., ['email'] or ['last_name', 'first_name'])
    - 'using': Optional index method (default: 'btree', other options include 'hash', 'gist', etc.)

Examples: [
    {"table": "users", "columns": ["email"], "using": "btree"},
    {"table": "orders", "columns": ["user_id", "created_at"]}
]
If there is no hypothetical index, you can pass an empty list.""",
        default=[],
    ),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """
    Explains the execution plan for a SQL query.

    Args:
        sql: The SQL query to explain
        analyze: When True, actually runs the query for real statistics
        hypothetical_indexes: Optional list of indexes to simulate
    """
    try:
        sql_driver = await get_sql_driver(database)
        explain_tool = ExplainPlanTool(sql_driver=sql_driver)
        result: ExplainPlanArtifact | ErrorResult | None = None

        # If hypothetical indexes are specified, check for HypoPG extension
        if hypothetical_indexes and len(hypothetical_indexes) > 0:
            if analyze:
                return format_error_response("Cannot use analyze and hypothetical indexes together")
            try:
                # Use the common utility function to check if hypopg is installed
                (
                    is_hypopg_installed,
                    hypopg_message,
                ) = await check_hypopg_installation_status(sql_driver)

                # If hypopg is not installed, return the message
                if not is_hypopg_installed:
                    return format_text_response(hypopg_message)

                # HypoPG is installed, proceed with explaining with hypothetical indexes
                result = await explain_tool.explain_with_hypothetical_indexes(sql, hypothetical_indexes)
            except Exception:
                raise  # Re-raise the original exception
        elif analyze:
            try:
                # Use EXPLAIN ANALYZE
                result = await explain_tool.explain_analyze(sql)
            except Exception:
                raise  # Re-raise the original exception
        else:
            try:
                # Use basic EXPLAIN
                result = await explain_tool.explain(sql)
            except Exception:
                raise  # Re-raise the original exception

        if result and isinstance(result, ExplainPlanArtifact):
            return format_text_response(result.to_text())
        else:
            error_message = "Error processing explain plan"
            if isinstance(result, ErrorResult):
                error_message = result.to_text()
            return format_error_response(error_message)
    except Exception as e:
        logger.error(f"Error explaining query: {e}")
        return format_error_response(str(e))


# Query function declaration without the decorator - we'll add it dynamically based on access mode
async def execute_sql(
    sql: str = Field(description="SQL to run", default="all"),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """Executes a SQL query against the database."""
    try:
        sql_driver = await get_sql_driver(database)
        rows = await sql_driver.execute_query(sql)  # type: ignore
        if rows is None:
            return format_text_response("No results")
        return format_text_response(list([r.cells for r in rows]))
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Analyze frequently executed queries in the database and recommend optimal indexes",
    annotations=ToolAnnotations(
        title="Analyze Workload Indexes",
        readOnlyHint=True,
    ),
)
@validate_call
async def analyze_workload_indexes(
    max_index_size_mb: int = Field(description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(description="Method to use for analysis", default="dta"),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """Analyze frequently executed queries in the database and recommend optimal indexes."""
    try:
        sql_driver = await get_sql_driver(database)
        if method == "dta":
            index_tuning = DatabaseTuningAdvisor(sql_driver)
        else:
            index_tuning = LLMOptimizerTool(sql_driver)
        dta_tool = TextPresentation(sql_driver, index_tuning)
        result = await dta_tool.analyze_workload(max_index_size_mb=max_index_size_mb)
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error analyzing workload: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Analyze a list of (up to 10) SQL queries and recommend optimal indexes",
    annotations=ToolAnnotations(
        title="Analyze Query Indexes",
        readOnlyHint=True,
    ),
)
@validate_call
async def analyze_query_indexes(
    queries: list[str] = Field(description="List of Query strings to analyze"),
    max_index_size_mb: int = Field(description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(description="Method to use for analysis", default="dta"),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """Analyze a list of SQL queries and recommend optimal indexes."""
    if len(queries) == 0:
        return format_error_response("Please provide a non-empty list of queries to analyze.")
    if len(queries) > MAX_NUM_INDEX_TUNING_QUERIES:
        return format_error_response(f"Please provide a list of up to {MAX_NUM_INDEX_TUNING_QUERIES} queries to analyze.")

    try:
        sql_driver = await get_sql_driver(database)
        if method == "dta":
            index_tuning = DatabaseTuningAdvisor(sql_driver)
        else:
            index_tuning = LLMOptimizerTool(sql_driver)
        dta_tool = TextPresentation(sql_driver, index_tuning)
        result = await dta_tool.analyze_queries(queries=queries, max_index_size_mb=max_index_size_mb)
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error analyzing queries: {e}")
        return format_error_response(str(e))


@mcp.tool(
    description="Analyzes database health. Here are the available health checks:\n"
    "- index - checks for invalid, duplicate, and bloated indexes\n"
    "- connection - checks the number of connection and their utilization\n"
    "- vacuum - checks vacuum health for transaction id wraparound\n"
    "- sequence - checks sequences at risk of exceeding their maximum value\n"
    "- replication - checks replication health including lag and slots\n"
    "- buffer - checks for buffer cache hit rates for indexes and tables\n"
    "- constraint - checks for invalid constraints\n"
    "- all - runs all checks\n"
    "You can optionally specify a single health check or a comma-separated list of health checks. The default is 'all' checks.",
    annotations=ToolAnnotations(
        title="Analyze Database Health",
        readOnlyHint=True,
    ),
)
async def analyze_db_health(
    health_type: str = Field(
        description=f"Optional. Valid values are: {', '.join(sorted([t.value for t in HealthType]))}.",
        default="all",
    ),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    """Analyze database health for specified components.

    Args:
        health_type: Comma-separated list of health check types to perform.
                    Valid values: index, connection, vacuum, sequence, replication, buffer, constraint, all
    """
    try:
        health_tool = DatabaseHealthTool(await get_sql_driver(database))
        result = await health_tool.health(health_type=health_type)
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error analyzing database health: {obfuscate_password(str(e))}")
        return format_error_response(str(e))


@mcp.tool(
    name="get_top_queries",
    description=f"Reports the slowest or most resource-intensive queries using data from the '{PG_STAT_STATEMENTS}' extension.",
    annotations=ToolAnnotations(
        title="Get Top Queries",
        readOnlyHint=True,
    ),
)
async def get_top_queries(
    sort_by: str = Field(
        description="Ranking criteria: 'total_time' for total execution time or 'mean_time' for mean execution time per call, or 'resources' "
        "for resource-intensive queries",
        default="resources",
    ),
    limit: int = Field(description="Number of queries to return when ranking based on mean_time or total_time", default=10),
    database: str | None = Field(description=DATABASE_PARAM_DESCRIPTION, default=None),
) -> ResponseType:
    try:
        sql_driver = await get_sql_driver(database)
        top_queries_tool = TopQueriesCalc(sql_driver=sql_driver)

        if sort_by == "resources":
            result = await top_queries_tool.get_top_resource_queries()
            return format_text_response(result)
        elif sort_by == "mean_time" or sort_by == "total_time":
            # Map the sort_by values to what get_top_queries_by_time expects
            result = await top_queries_tool.get_top_queries_by_time(limit=limit, sort_by="mean" if sort_by == "mean_time" else "total")
        else:
            return format_error_response("Invalid sort criteria. Please use 'resources' or 'mean_time' or 'total_time'.")
        return format_text_response(result)
    except Exception as e:
        logger.error(f"Error getting slow queries: {e}")
        return format_error_response(str(e))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _aliases_from_env() -> dict[str, str]:
    """Collect DATABASE_URI_<ALIAS> environment variables as {alias: uri}."""
    return {
        key[len(ALIAS_ENV_PREFIX) :].lower(): value
        for key, value in os.environ.items()
        if key.startswith(ALIAS_ENV_PREFIX) and len(key) > len(ALIAS_ENV_PREFIX) and value
    }


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument(
        "database_url",
        help="Default database connection URL (optional; used when a tool call does not name a database)",
        nargs="?",
    )
    parser.add_argument(
        "--database",
        dest="databases",
        action="append",
        default=[],
        metavar="ALIAS=URI",
        help="Register a named database that tools can target via their 'database' argument. "
        f"Repeatable. Can also be set with {ALIAS_ENV_PREFIX}<ALIAS> environment variables.",
    )
    parser.add_argument(
        "--allowed-hosts",
        type=str,
        default=os.environ.get("POSTGRES_MCP_ALLOWED_HOSTS"),
        help="Comma-separated list of hosts that connection URIs passed in tool calls or the X-Database-URI header "
        "may point to (env: POSTGRES_MCP_ALLOWED_HOSTS). Default: any host. Configured aliases are always allowed.",
    )
    parser.add_argument(
        "--no-dynamic-uri",
        action="store_true",
        default=_env_flag("POSTGRES_MCP_NO_DYNAMIC_URI"),
        help="Reject connection URIs in tool calls and headers; only the default database and configured aliases "
        "are usable (env: POSTGRES_MCP_NO_DYNAMIC_URI=1)",
    )
    parser.add_argument(
        "--max-pools",
        type=int,
        default=int(os.environ.get("POSTGRES_MCP_MAX_POOLS", DEFAULT_MAX_POOLS)),
        help=f"Maximum number of cached connection pools, one per database (default: {DEFAULT_MAX_POOLS})",
    )
    parser.add_argument(
        "--pool-idle-ttl",
        type=float,
        default=float(os.environ.get("POSTGRES_MCP_POOL_IDLE_TTL", DEFAULT_IDLE_TTL_SECONDS)),
        help=f"Seconds after which an unused non-default pool is closed; 0 disables (default: {DEFAULT_IDLE_TTL_SECONDS:g})",
    )
    parser.add_argument(
        "--access-mode",
        type=str,
        choices=[mode.value for mode in AccessMode],
        default=AccessMode.UNRESTRICTED.value,
        help="Set SQL access mode: unrestricted (unrestricted) or restricted (read-only with protections)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Select MCP transport: stdio (default), sse, or streamable-http",
    )
    parser.add_argument(
        "--sse-host",
        type=str,
        default="localhost",
        help="Host to bind SSE server to (default: localhost)",
    )
    parser.add_argument(
        "--sse-port",
        type=int,
        default=8000,
        help="Port for SSE server (default: 8000)",
    )
    parser.add_argument(
        "--streamable-http-host",
        type=str,
        default="localhost",
        help="Host to bind streamable HTTP server to (default: localhost)",
    )
    parser.add_argument(
        "--streamable-http-port",
        type=int,
        default=8000,
        help="Port for streamable HTTP server (default: 8000)",
    )

    args = parser.parse_args()

    # Store the access mode in the global variable
    global current_access_mode
    current_access_mode = AccessMode(args.access_mode)

    # Add the query tool with a description and annotations appropriate to the access mode
    if current_access_mode == AccessMode.UNRESTRICTED:
        mcp.add_tool(
            execute_sql,
            description="Execute any SQL query",
            annotations=ToolAnnotations(
                title="Execute SQL",
                destructiveHint=True,
            ),
        )
    else:
        mcp.add_tool(
            execute_sql,
            description="Execute a read-only SQL query",
            annotations=ToolAnnotations(
                title="Execute SQL (Read-Only)",
                readOnlyHint=True,
            ),
        )

    logger.info(f"Starting PostgreSQL MCP Server in {current_access_mode.upper()} mode")

    # Get the default database URL from environment variable or command line (optional)
    database_url = os.environ.get("DATABASE_URI", args.database_url) or None

    aliases = _aliases_from_env()
    for item in args.databases:
        name, sep, uri = item.partition("=")
        if not sep or not name.strip() or not uri.strip():
            raise ValueError(f"Invalid --database value {obfuscate_password(item)!r}: expected ALIAS=URI")
        aliases[name.strip().lower()] = uri.strip()

    db_registry.configure(
        default_uri=database_url,
        aliases=aliases,
        allow_dynamic_uri=not args.no_dynamic_uri,
        allowed_hosts=args.allowed_hosts.split(",") if args.allowed_hosts else None,
        max_pools=args.max_pools,
        idle_ttl=args.pool_idle_ttl,
    )

    if aliases:
        logger.info(f"Configured database aliases: {', '.join(sorted(aliases))}")
    if not database_url and not aliases and args.no_dynamic_uri:
        raise ValueError(
            "Error: No database configured. Provide a default database ('DATABASE_URI' or command-line argument), "
            "aliases (--database ALIAS=URI), or allow connection URIs in tool calls.",
        )

    if database_url:
        # Connect the default pool eagerly so configuration problems show up at startup
        try:
            await db_registry.get_pool(database_url)
            logger.info("Successfully connected to default database and initialized connection pool")
        except Exception as e:
            logger.warning(
                f"Could not connect to default database: {obfuscate_password(str(e))}",
            )
            logger.warning(
                "The MCP server will start but operations on the default database will fail until it is reachable.",
            )
    else:
        logger.info("No default database configured: tool calls must name a database (argument or X-Database-URI header)")

    # Set up proper shutdown handling
    try:
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        logger.warning("Signal handling not supported on Windows")
        pass

    # Run the server with the selected transport (always async)
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        mcp.settings.host = args.sse_host
        mcp.settings.port = args.sse_port
        await mcp.run_sse_async()
    elif args.transport == "streamable-http":
        mcp.settings.host = args.streamable_http_host
        mcp.settings.port = args.streamable_http_port
        await mcp.run_streamable_http_async()


async def shutdown(sig=None):
    """Clean shutdown of the server."""
    global shutdown_in_progress

    if shutdown_in_progress:
        logger.warning("Forcing immediate exit")
        # Use sys.exit instead of os._exit to allow for proper cleanup
        sys.exit(1)

    shutdown_in_progress = True

    if sig:
        logger.info(f"Received exit signal {sig.name}")

    # Close database connections
    try:
        await db_registry.close_all()
        logger.info("Closed database connections")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    # Exit with appropriate status code
    sys.exit(128 + sig if sig is not None else 0)

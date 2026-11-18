import asyncio
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from postgres_mcp.sql import DatabaseRegistry
from postgres_mcp.sql import DatabaseResolutionError
from postgres_mcp.sql import SqlDriver
from postgres_mcp.sql.db_registry import parse_env_file
from postgres_mcp.sql.extension_utils import get_postgres_version
from postgres_mcp.sql.extension_utils import reset_postgres_version_cache

SHOP = "postgresql://user:secret@shop-host:5432/shop"
CRM = "postgresql://user:secret@crm-host:5432/crm"
LEGACY = "postgresql://user:secret@legacy-host:5432/main"

ENV = {"DATABASE_URI_SHOP": SHOP, "DATABASE_URI_CRM": CRM, "PATH": "/usr/bin", "DATABASE_URI_EMPTY": " "}

_real_probe = DatabaseRegistry._probe


def make(environ=None, **kwargs) -> DatabaseRegistry:
    return DatabaseRegistry(environ=ENV if environ is None else environ, **kwargs)


@pytest.fixture(autouse=True)
def no_probe():
    """Skip the real probe connection in unit tests."""
    with patch.object(DatabaseRegistry, "_probe", AsyncMock()):
        yield


# --------------------------------------------------------------------------- configuration


def test_databases_come_from_prefixed_env():
    reg = make()
    assert reg.databases() == {"shop": SHOP, "crm": CRM}


def test_custom_prefix():
    reg = make({"PGMCP_DB_ANALYTICS": SHOP, "DATABASE_URI_CRM": CRM}, prefix="PGMCP_DB_")
    assert reg.databases() == {"analytics": SHOP}


def test_legacy_database_uri_is_default():
    reg = make(fallback_default_uri=LEGACY)
    assert reg.databases()["default"] == LEGACY
    assert reg.default_database() == "default"
    assert reg.resolve() == ("default", LEGACY)


def test_single_database_is_default():
    reg = make({"DATABASE_URI_SHOP": SHOP})
    assert reg.resolve() == ("shop", SHOP)


def test_explicit_default_name():
    reg = make(default_name="CRM")
    assert reg.resolve() == ("crm", CRM)
    assert make(default_name="missing").default_database() is None


def test_no_default_with_several_databases():
    reg = make()
    with pytest.raises(DatabaseResolutionError, match=r"no default database.*one of: crm, shop"):
        reg.resolve()


def test_nothing_configured():
    with pytest.raises(DatabaseResolutionError, match=r"No databases are configured.*DATABASE_URI_<NAME>"):
        make({}).resolve()


def test_env_is_read_on_every_lookup():
    env = {"DATABASE_URI_SHOP": SHOP}
    reg = make(env)
    assert reg.databases() == {"shop": SHOP}
    env["DATABASE_URI_CRM"] = CRM
    assert reg.resolve("crm") == ("crm", CRM)


def test_parse_env_file():
    text = """
    # comment
    DATABASE_URI_A=postgresql://a/a
    export DATABASE_URI_B = "postgresql://b/b"
    DATABASE_URI_C='host=c dbname=c'
    garbage line
    """
    assert parse_env_file(text) == {
        "DATABASE_URI_A": "postgresql://a/a",
        "DATABASE_URI_B": "postgresql://b/b",
        "DATABASE_URI_C": "host=c dbname=c",
    }


def test_env_file_is_reloaded_on_change(tmp_path):
    env_file = tmp_path / "dbs.env"
    env_file.write_text(f"DATABASE_URI_SHOP={SHOP}\n")
    reg = make({}, env_file=str(env_file))
    assert reg.databases() == {"shop": SHOP}

    env_file.write_text(f"DATABASE_URI_SHOP={SHOP}\nDATABASE_URI_CRM={CRM}\n")
    os.utime(env_file, (1, 1))  # make sure mtime changes even on coarse filesystems
    assert reg.resolve("crm") == ("crm", CRM)

    env_file.unlink()
    assert reg.databases() == {}


def test_env_file_overrides_process_env(tmp_path):
    env_file = tmp_path / "dbs.env"
    env_file.write_text(f"DATABASE_URI_SHOP={LEGACY}\n")
    reg = make(env_file=str(env_file))
    assert reg.databases() == {"shop": LEGACY, "crm": CRM}


def test_missing_env_file_is_not_an_error(tmp_path):
    reg = make(env_file=str(tmp_path / "nope.env"))
    assert reg.databases() == {"shop": SHOP, "crm": CRM}


# --------------------------------------------------------------------------- resolution


def test_resolve_by_name_is_case_insensitive():
    reg = make()
    assert reg.resolve("CRM") == ("crm", CRM)
    assert reg.resolve(" shop ") == ("shop", SHOP)


def test_tool_argument_wins_over_header():
    reg = make()
    assert reg.resolve("shop", "crm") == ("shop", SHOP)
    assert reg.resolve(None, "crm") == ("crm", CRM)
    assert reg.resolve("", "crm") == ("crm", CRM)


def test_unknown_database():
    with pytest.raises(DatabaseResolutionError, match=r"Unknown database 'nope'.*crm, shop"):
        make().resolve("nope")


@pytest.mark.parametrize(
    "value",
    [
        "postgresql://user:hunter2@evil-host/db",
        "postgres://evil-host/db",
        "host=evil-host dbname=db password=hunter2",
        "user:hunter2@evil-host",
    ],
)
def test_connection_strings_are_rejected(value):
    reg = make()
    for args in ((value, None), (None, value)):
        with pytest.raises(DatabaseResolutionError, match="Connection strings are not accepted") as exc:
            reg.resolve(*args)
        assert "hunter2" not in str(exc.value)
        assert "evil-host" not in str(exc.value)


def test_describe_has_no_credentials():
    reg = make(default_name="shop")
    info = reg.describe()
    assert "secret" not in str(info)
    assert info == [
        {"name": "crm", "target": "crm-host:5432/crm", "default": False},
        {"name": "shop", "target": "shop-host:5432/shop", "default": True},
    ]


# --------------------------------------------------------------------------- pools


def _fake_pools():
    """Patch DbConnPool so every instance is a mock that 'connects' instantly."""
    created: list[MagicMock] = []

    def factory(uri):
        pool = MagicMock()
        pool.connection_url = uri
        pool.pool_connect = AsyncMock()
        pool.close = AsyncMock()
        created.append(pool)
        return pool

    return created, patch("postgres_mcp.sql.db_registry.DbConnPool", side_effect=factory)


def _hosts(n):
    return {f"DATABASE_URI_H{i}": f"postgresql://h{i}/db" for i in range(1, n + 1)}


@pytest.mark.asyncio
async def test_pool_is_cached_per_database():
    created, p = _fake_pools()
    with p:
        reg = make()
        a1 = await reg.get_pool(SHOP)
        a2 = await reg.get_pool(SHOP)
        b = await reg.get_pool(CRM)
    assert a1 is a2
    assert a1 is not b
    assert len(created) == 2


@pytest.mark.asyncio
async def test_concurrent_requests_create_one_pool():
    created: list[MagicMock] = []

    async def slow_connect(*_args, **_kwargs):
        await asyncio.sleep(0.01)

    def factory(uri):
        pool = MagicMock()
        pool.pool_connect = AsyncMock(side_effect=slow_connect)
        created.append(pool)
        return pool

    with patch("postgres_mcp.sql.db_registry.DbConnPool", side_effect=factory):
        reg = make()
        pools = await asyncio.gather(*(reg.get_pool(CRM) for _ in range(10)))
    assert len(created) == 1
    assert all(pl is pools[0] for pl in pools)


@pytest.mark.asyncio
async def test_failed_connect_is_not_cached():
    attempts = []

    def factory(uri):
        pool = MagicMock()
        pool.pool_connect = AsyncMock(side_effect=ValueError("Connection attempt failed"))
        attempts.append(pool)
        return pool

    with patch("postgres_mcp.sql.db_registry.DbConnPool", side_effect=factory):
        reg = make()
        for _ in range(2):
            with pytest.raises(ValueError, match="Connection attempt failed"):
                await reg.get_pool(CRM)
    assert len(attempts) == 2
    assert reg.pool_count == 0


@pytest.mark.asyncio
async def test_lru_eviction_keeps_default_pool():
    created, p = _fake_pools()
    with p:
        reg = make(_hosts(3), default_name="h1", max_pools=2)
        default_pool = await reg.get_pool("postgresql://h1/db")
        second = await reg.get_pool("postgresql://h2/db")
        await reg.get_pool("postgresql://h3/db")  # evicts h2 (LRU, not pinned)
        await asyncio.sleep(0)  # let background close run
        assert reg.pool_count == 2
        second.close.assert_awaited_once()
        default_pool.close.assert_not_called()
        assert await reg.get_pool("postgresql://h1/db") is default_pool


@pytest.mark.asyncio
async def test_lru_order_is_updated_on_access():
    created, p = _fake_pools()
    with p:
        reg = make(_hosts(3), max_pools=2)
        p1 = await reg.get_pool("postgresql://h1/db")
        p2 = await reg.get_pool("postgresql://h2/db")
        await reg.get_pool("postgresql://h1/db")  # h1 becomes most recent
        await reg.get_pool("postgresql://h3/db")  # evicts h2
        await asyncio.sleep(0)
        p2.close.assert_awaited_once()
        p1.close.assert_not_called()


@pytest.mark.asyncio
async def test_idle_ttl_eviction():
    created, p = _fake_pools()
    clock = [1000.0]
    with p, patch("postgres_mcp.sql.db_registry.time.monotonic", side_effect=lambda: clock[0]):
        reg = make(default_name="shop", idle_ttl=60)
        default_pool = await reg.get_pool(SHOP)
        idle = await reg.get_pool(CRM)
        clock[0] += 61
        await reg.get_pool(SHOP)  # any access triggers the sweep
        await asyncio.sleep(0)
        idle.close.assert_awaited_once()
        default_pool.close.assert_not_called()
        assert reg.pool_count == 1
        assert await reg.get_pool(CRM) is not idle
    assert len(created) == 3


@pytest.mark.asyncio
async def test_pool_closed_when_database_removed_from_config():
    created, p = _fake_pools()
    env = dict(ENV)
    with p:
        reg = make(env)
        crm_pool = await reg.get_pool(CRM)
        del env["DATABASE_URI_CRM"]
        await reg.get_pool(SHOP)
        await asyncio.sleep(0)
        crm_pool.close.assert_awaited_once()
        assert reg.pool_count == 1


@pytest.mark.asyncio
async def test_close_all():
    created, p = _fake_pools()
    with p:
        reg = make()
        await reg.get_pool(SHOP)
        await reg.get_pool(CRM)
        await reg.close_all()
    assert reg.pool_count == 0
    for pool in created:
        pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_failure_is_readable_and_hides_password():
    reg = make()
    with patch(
        "postgres_mcp.sql.db_registry.psycopg.AsyncConnection.connect",
        AsyncMock(side_effect=Exception('password authentication failed for user "u" (postgresql://u:hunter2@h/db)')),
    ):
        with pytest.raises(ValueError) as exc:
            await _real_probe(reg, "postgresql://u:hunter2@h:5432/db")
    assert "password authentication failed" in str(exc.value)
    assert "h:5432/db" in str(exc.value)
    assert "hunter2" not in str(exc.value)


# --------------------------------------------------------------------------- server wiring


@pytest.mark.asyncio
async def test_get_sql_driver_routes_by_argument_and_header():
    from postgres_mcp import server

    reg = make(default_name="shop")
    pools = {SHOP: MagicMock(name="shop"), CRM: MagicMock(name="crm")}
    reg.get_pool = AsyncMock(side_effect=lambda uri: pools[uri])  # type: ignore[method-assign]

    with patch.object(server, "db_registry", reg), patch.object(server, "_database_from_request_headers", return_value=None):
        assert (await server.get_sql_driver()).conn is pools[SHOP]
        assert (await server.get_sql_driver("crm")).conn is pools[CRM]

    with patch.object(server, "db_registry", reg), patch.object(server, "_database_from_request_headers", return_value="crm"):
        assert (await server.get_sql_driver()).conn is pools[CRM]
        assert (await server.get_sql_driver("shop")).conn is pools[SHOP]


@pytest.mark.asyncio
async def test_tools_reject_connection_strings_and_unknown_names():
    from postgres_mcp import server

    with patch.object(server, "db_registry", make()), patch.object(server, "_database_from_request_headers", return_value=None):
        result = await server.list_schemas(database="postgresql://u:hunter2@evil/db")
        assert "Connection strings are not accepted" in result[0].text
        assert "hunter2" not in result[0].text
        result = await server.list_schemas(database="missing")
        assert "Unknown database 'missing'" in result[0].text
        result = await server.list_schemas(database=None)
        assert "no default database" in result[0].text


@pytest.mark.asyncio
async def test_list_databases_tool():
    from postgres_mcp import server

    with patch.object(server, "db_registry", make(default_name="crm")):
        text = (await server.list_databases())[0].text
    assert "'name': 'crm'" in text and "'name': 'shop'" in text
    assert "secret" not in text

    with patch.object(server, "db_registry", make({})):
        text = (await server.list_databases())[0].text
    assert "No databases are configured" in text


def test_header_lookup_outside_request_returns_none():
    from postgres_mcp import server

    assert server._database_from_request_headers() is None


@pytest.mark.asyncio
async def test_header_lookup_reads_http_request():
    from postgres_mcp import server

    ctx = MagicMock()
    ctx.request_context.request.headers = {"x-database": "crm"}
    with patch.object(server.mcp, "get_context", return_value=ctx):
        assert server._database_from_request_headers() == "crm"


def test_tools_expose_database_parameter():
    from postgres_mcp import server

    tools = {t.name: t for t in server.mcp._tool_manager.list_tools()}
    for name in (
        "list_schemas",
        "list_objects",
        "get_object_details",
        "explain_query",
        "analyze_workload_indexes",
        "analyze_query_indexes",
        "analyze_db_health",
        "get_top_queries",
    ):
        assert "database" in tools[name].parameters["properties"], name
        assert "database" not in tools[name].parameters.get("required", []), name
    assert "list_databases" in tools


# --------------------------------------------------------------------------- per-database version cache


@pytest.mark.asyncio
async def test_postgres_version_cached_per_database():
    reset_postgres_version_cache()

    def driver_for(uri, version):
        conn = MagicMock()
        conn.connection_url = uri
        driver = SqlDriver(conn=conn)
        driver.execute_query = AsyncMock(return_value=[SqlDriver.RowResult(cells={"server_version": version})])  # type: ignore[method-assign]
        return driver

    d12 = driver_for(SHOP, "12.18")
    d16 = driver_for(CRM, "16.4")
    try:
        assert await get_postgres_version(d12) == 12
        assert await get_postgres_version(d16) == 16
        assert await get_postgres_version(d12) == 12
        d12.execute_query.assert_awaited_once()  # type: ignore[attr-defined]
    finally:
        reset_postgres_version_cache()

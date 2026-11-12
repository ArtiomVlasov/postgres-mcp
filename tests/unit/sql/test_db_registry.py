import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from postgres_mcp.sql import DatabaseRegistry
from postgres_mcp.sql import DatabaseResolutionError
from postgres_mcp.sql import SqlDriver
from postgres_mcp.sql.extension_utils import get_postgres_version
from postgres_mcp.sql.extension_utils import reset_postgres_version_cache

DEFAULT = "postgresql://user:secret@default-host:5432/main"
ANALYTICS = "postgresql://user:secret@analytics-host:5432/analytics"
OTHER = "postgresql://user:secret@other-host:5432/other"

_real_probe = DatabaseRegistry._probe


# --------------------------------------------------------------------------- resolution


def test_resolve_falls_back_to_default():
    reg = DatabaseRegistry(default_uri=DEFAULT)
    assert reg.resolve() == DEFAULT
    assert reg.resolve(None, None) == DEFAULT
    assert reg.resolve("  ", "") == DEFAULT


def test_resolve_alias_is_case_insensitive():
    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"Analytics": ANALYTICS})
    assert reg.resolve("analytics") == ANALYTICS
    assert reg.resolve("ANALYTICS") == ANALYTICS
    assert reg.resolve("default") == DEFAULT


def test_resolve_uri_passthrough():
    reg = DatabaseRegistry(default_uri=DEFAULT)
    assert reg.resolve(OTHER) == OTHER
    kv = "host=other-host dbname=other user=u password=p"
    assert reg.resolve(kv) == kv


def test_tool_argument_wins_over_header():
    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"analytics": ANALYTICS})
    assert reg.resolve("analytics", OTHER) == ANALYTICS
    assert reg.resolve(None, OTHER) == OTHER
    assert reg.resolve(None, "analytics") == ANALYTICS


def test_no_default_and_nothing_requested():
    reg = DatabaseRegistry(aliases={"analytics": ANALYTICS})
    with pytest.raises(DatabaseResolutionError, match="no default database"):
        reg.resolve()


def test_unknown_alias():
    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"analytics": ANALYTICS})
    with pytest.raises(DatabaseResolutionError, match=r"Unknown database alias 'nope'.*analytics, default"):
        reg.resolve("nope")


def test_dynamic_uri_can_be_disabled():
    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"analytics": ANALYTICS}, allow_dynamic_uri=False)
    assert reg.resolve("analytics") == ANALYTICS
    with pytest.raises(DatabaseResolutionError, match="disabled"):
        reg.resolve(OTHER)
    with pytest.raises(DatabaseResolutionError, match="disabled"):
        reg.resolve(None, OTHER)


def test_allowed_hosts():
    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"analytics": ANALYTICS}, allowed_hosts=["Other-Host", " localhost "])
    assert reg.resolve(OTHER) == OTHER
    assert reg.resolve("postgresql:///local_db") == "postgresql:///local_db"  # unix socket -> localhost
    # aliases are operator-configured and bypass the allowlist
    assert reg.resolve("analytics") == ANALYTICS
    with pytest.raises(DatabaseResolutionError, match=r"not allowed: evil\.example\.com"):
        reg.resolve("postgresql://u:p@evil.example.com/db")
    with pytest.raises(DatabaseResolutionError, match=r"not allowed: evil\.example\.com"):
        reg.resolve("postgresql://u:p@other-host,evil.example.com/db")


def test_invalid_uri_is_rejected_without_leaking_password():
    reg = DatabaseRegistry(default_uri=DEFAULT)
    with pytest.raises(DatabaseResolutionError) as exc:
        reg.resolve("host=x password=topsecret bogus")
    assert "topsecret" not in str(exc.value)


def test_describe_has_no_credentials():
    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"analytics": ANALYTICS}, allowed_hosts=["a", "b"])
    info = reg.describe()
    assert "secret" not in str(info)
    assert info["databases"][0] == {"alias": "default", "target": "default-host:5432/main", "default": True}
    assert info["databases"][1]["alias"] == "analytics"
    assert info["allowed_hosts"] == ["a", "b"]
    assert info["connection_strings_allowed"] is True


# --------------------------------------------------------------------------- pools


def _fake_pool_factory():
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


@pytest.fixture(autouse=True)
def no_probe():
    """Skip the real probe connection in unit tests."""
    with patch.object(DatabaseRegistry, "_probe", AsyncMock()):
        yield


@pytest.mark.asyncio
async def test_pool_is_cached_per_uri():
    created, p = _fake_pool_factory()
    with p:
        reg = DatabaseRegistry(default_uri=DEFAULT)
        a1 = await reg.get_pool(DEFAULT)
        a2 = await reg.get_pool(DEFAULT)
        b = await reg.get_pool(OTHER)
    assert a1 is a2
    assert a1 is not b
    assert len(created) == 2
    assert reg.pool_count == 2


@pytest.mark.asyncio
async def test_concurrent_requests_create_one_pool():
    created, p = _fake_pool_factory()

    async def slow_connect(*_args, **_kwargs):
        await asyncio.sleep(0.01)

    with p:
        reg = DatabaseRegistry()
        # make the first created pool slow to connect
        orig_side_effect = p.kwargs["side_effect"]

        def slow_factory(uri):
            pool = orig_side_effect(uri)
            pool.pool_connect = AsyncMock(side_effect=slow_connect)
            return pool

        with patch("postgres_mcp.sql.db_registry.DbConnPool", side_effect=slow_factory):
            pools = await asyncio.gather(*(reg.get_pool(OTHER) for _ in range(10)))
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
        reg = DatabaseRegistry()
        for _ in range(2):
            with pytest.raises(ValueError, match="Connection attempt failed"):
                await reg.get_pool(OTHER)
    assert len(attempts) == 2
    assert reg.pool_count == 0


@pytest.mark.asyncio
async def test_lru_eviction_keeps_default_pool():
    created, p = _fake_pool_factory()
    with p:
        reg = DatabaseRegistry(default_uri=DEFAULT, max_pools=2)
        default_pool = await reg.get_pool(DEFAULT)
        first = await reg.get_pool("postgresql://h1/db")
        await reg.get_pool("postgresql://h2/db")  # evicts h1 (LRU, not pinned)
        await asyncio.sleep(0)  # let background close run
        assert reg.pool_count == 2
        first.close.assert_awaited_once()
        default_pool.close.assert_not_called()
        # default is still cached
        assert await reg.get_pool(DEFAULT) is default_pool
    assert len(created) == 3


@pytest.mark.asyncio
async def test_lru_order_is_updated_on_access():
    created, p = _fake_pool_factory()
    with p:
        reg = DatabaseRegistry(max_pools=2)
        p1 = await reg.get_pool("postgresql://h1/db")
        p2 = await reg.get_pool("postgresql://h2/db")
        await reg.get_pool("postgresql://h1/db")  # h1 becomes most recent
        await reg.get_pool("postgresql://h3/db")  # evicts h2
        await asyncio.sleep(0)
        p2.close.assert_awaited_once()
        p1.close.assert_not_called()


@pytest.mark.asyncio
async def test_idle_ttl_eviction():
    created, p = _fake_pool_factory()
    clock = [1000.0]
    with p, patch("postgres_mcp.sql.db_registry.time.monotonic", side_effect=lambda: clock[0]):
        reg = DatabaseRegistry(default_uri=DEFAULT, idle_ttl=60)
        default_pool = await reg.get_pool(DEFAULT)
        idle = await reg.get_pool(OTHER)
        clock[0] += 61
        await reg.get_pool(DEFAULT)  # any access triggers the sweep
        await asyncio.sleep(0)
        idle.close.assert_awaited_once()
        default_pool.close.assert_not_called()
        assert reg.pool_count == 1
        # a new request for the evicted database creates a fresh pool
        assert await reg.get_pool(OTHER) is not idle
    assert len(created) == 3


@pytest.mark.asyncio
async def test_close_all():
    created, p = _fake_pool_factory()
    with p:
        reg = DatabaseRegistry(default_uri=DEFAULT)
        await reg.get_pool(DEFAULT)
        await reg.get_pool(OTHER)
        await reg.close_all()
    assert reg.pool_count == 0
    for pool in created:
        pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_failure_is_readable_and_hides_password():
    reg = DatabaseRegistry()
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

    reg = DatabaseRegistry(default_uri=DEFAULT, aliases={"analytics": ANALYTICS})
    pools = {DEFAULT: MagicMock(name="default"), ANALYTICS: MagicMock(name="analytics"), OTHER: MagicMock(name="other")}
    reg.get_pool = AsyncMock(side_effect=lambda uri: pools[uri])  # type: ignore[method-assign]

    with patch.object(server, "db_registry", reg), patch.object(server, "_database_from_request_headers", return_value=None):
        assert (await server.get_sql_driver()).conn is pools[DEFAULT]
        assert (await server.get_sql_driver("analytics")).conn is pools[ANALYTICS]
        assert (await server.get_sql_driver(OTHER)).conn is pools[OTHER]

    with patch.object(server, "db_registry", reg), patch.object(server, "_database_from_request_headers", return_value=OTHER):
        assert (await server.get_sql_driver()).conn is pools[OTHER]
        assert (await server.get_sql_driver("analytics")).conn is pools[ANALYTICS]


@pytest.mark.asyncio
async def test_tool_reports_resolution_error():
    from postgres_mcp import server

    reg = DatabaseRegistry()  # no default database
    with patch.object(server, "db_registry", reg):
        result = await server.list_schemas(database=None)
        assert "no default database" in result[0].text
        result = await server.list_schemas(database="missing")
        assert "Unknown database alias 'missing'" in result[0].text


def test_header_lookup_outside_request_returns_none():
    from postgres_mcp import server

    assert server._database_from_request_headers() is None


@pytest.mark.asyncio
async def test_header_lookup_reads_http_request():
    from postgres_mcp import server

    ctx = MagicMock()
    ctx.request_context.request.headers = {"x-database-uri": OTHER}
    with patch.object(server.mcp, "get_context", return_value=ctx):
        assert server._database_from_request_headers() == OTHER


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

    d12 = driver_for(DEFAULT, "12.18")
    d16 = driver_for(OTHER, "16.4")
    try:
        assert await get_postgres_version(d12) == 12
        assert await get_postgres_version(d16) == 16
        assert await get_postgres_version(d12) == 12
        d12.execute_query.assert_awaited_once()  # type: ignore[attr-defined]
    finally:
        reset_postgres_version_cache()

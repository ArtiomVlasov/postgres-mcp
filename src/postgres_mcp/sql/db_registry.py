"""Registry of connection pools for serving several databases from one server.

A single server instance can work with many databases. The database for a
tool call is resolved in this order:

1. the ``database`` argument of the tool (an alias or a connection URI),
2. the ``X-Database-URI`` HTTP header (sse / streamable-http transports only),
3. the default database the server was started with (``DATABASE_URI``).

Pools are created lazily, cached per connection string, and evicted when
idle for too long or when the cache is full (least recently used first).
The default pool is never evicted.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable
from typing import Optional

import psycopg
from psycopg.conninfo import conninfo_to_dict

from .sql_driver import DbConnPool
from .sql_driver import obfuscate_password

logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "default"
DEFAULT_MAX_POOLS = 10
DEFAULT_IDLE_TTL_SECONDS = 600.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10


class DatabaseResolutionError(ValueError):
    """Raised when the database for a request cannot be determined or is not allowed."""


def looks_like_conninfo(value: str) -> bool:
    """Whether a value is a connection string rather than an alias name."""
    return "://" in value or "=" in value


def conninfo_hosts(uri: str) -> list[str]:
    """Return the hosts a connection string points to ('localhost' for unix sockets / no host)."""
    try:
        params = conninfo_to_dict(uri)
    except Exception as e:
        raise DatabaseResolutionError(f"Invalid connection string: {obfuscate_password(str(e))}") from e
    host = params.get("host") or params.get("hostaddr") or ""
    hosts = [h.strip().lower() for h in str(host).split(",") if h.strip()]
    return [("localhost" if h.startswith("/") else h) for h in hosts] or ["localhost"]


def describe_conninfo(uri: str) -> str:
    """Human-readable description of a connection string without credentials."""
    try:
        params = conninfo_to_dict(uri)
    except Exception:
        return obfuscate_password(uri) or ""
    host = params.get("host") or "localhost"
    port = params.get("port") or "5432"
    dbname = params.get("dbname") or params.get("user") or ""
    return f"{host}:{port}/{dbname}"


@dataclass
class _PoolEntry:
    pool: DbConnPool
    last_used: float = field(default_factory=time.monotonic)
    pinned: bool = False


class DatabaseRegistry:
    """Resolves which database a request targets and hands out cached pools."""

    def __init__(
        self,
        default_uri: Optional[str] = None,
        aliases: Optional[dict[str, str]] = None,
        allow_dynamic_uri: bool = True,
        allowed_hosts: Optional[Iterable[str]] = None,
        max_pools: int = DEFAULT_MAX_POOLS,
        idle_ttl: float = DEFAULT_IDLE_TTL_SECONDS,
    ):
        self.configure(
            default_uri=default_uri,
            aliases=aliases,
            allow_dynamic_uri=allow_dynamic_uri,
            allowed_hosts=allowed_hosts,
            max_pools=max_pools,
            idle_ttl=idle_ttl,
        )
        self._pools: OrderedDict[str, _PoolEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._closing: set[asyncio.Task[None]] = set()

    def configure(
        self,
        default_uri: Optional[str] = None,
        aliases: Optional[dict[str, str]] = None,
        allow_dynamic_uri: bool = True,
        allowed_hosts: Optional[Iterable[str]] = None,
        max_pools: int = DEFAULT_MAX_POOLS,
        idle_ttl: float = DEFAULT_IDLE_TTL_SECONDS,
    ) -> None:
        self.default_uri = default_uri or None
        self.aliases = {k.lower(): v for k, v in (aliases or {}).items()}
        self.allow_dynamic_uri = allow_dynamic_uri
        hosts = {h.strip().lower() for h in (allowed_hosts or []) if h and h.strip()}
        self.allowed_hosts: Optional[set[str]] = hosts or None
        self.max_pools = max(1, max_pools)
        self.idle_ttl = idle_ttl

    # ------------------------------------------------------------------ resolution

    def resolve(self, database: Optional[str] = None, header_value: Optional[str] = None) -> str:
        """Pick the connection string for a request.

        ``database`` (tool argument) wins over ``header_value`` (HTTP header),
        which wins over the default database.
        """
        for value, source in ((database, "tool argument 'database'"), (header_value, "X-Database-URI header")):
            if isinstance(value, str) and value.strip():
                return self._resolve_value(value.strip(), source)

        if self.default_uri:
            return self.default_uri

        known = ", ".join(sorted(self.aliases)) or "none"
        raise DatabaseResolutionError(
            "No database specified and the server has no default database. "
            "Pass 'database' (a configured alias or a postgresql:// URI). "
            f"Configured aliases: {known}."
        )

    def _resolve_value(self, value: str, source: str) -> str:
        alias = value.lower()
        if alias == DEFAULT_ALIAS and DEFAULT_ALIAS not in self.aliases:
            if not self.default_uri:
                raise DatabaseResolutionError("The server has no default database configured.")
            return self.default_uri
        if alias in self.aliases:
            return self.aliases[alias]

        if not looks_like_conninfo(value):
            known = ", ".join(sorted(self.aliases | ({DEFAULT_ALIAS: ""} if self.default_uri else {}))) or "none"
            raise DatabaseResolutionError(f"Unknown database alias '{value}' ({source}). Known aliases: {known}.")

        if not self.allow_dynamic_uri:
            raise DatabaseResolutionError(
                f"Connection strings in the {source} are disabled on this server. Use one of the configured aliases (see list_databases)."
            )
        if self.allowed_hosts is not None:
            denied = [h for h in conninfo_hosts(value) if h not in self.allowed_hosts]
            if denied:
                raise DatabaseResolutionError(f"Host(s) not allowed: {', '.join(denied)}. Allowed hosts: {', '.join(sorted(self.allowed_hosts))}.")
        else:
            # Still validate the string so errors come back early and readable.
            conninfo_hosts(value)
        return value

    def describe(self) -> dict[str, Any]:
        """Configuration summary safe to show to a model (no credentials)."""
        databases = []
        if self.default_uri:
            databases.append({"alias": DEFAULT_ALIAS, "target": describe_conninfo(self.default_uri), "default": True})
        for alias, uri in sorted(self.aliases.items()):
            databases.append({"alias": alias, "target": describe_conninfo(uri), "default": False})
        return {
            "databases": databases,
            "connection_strings_allowed": self.allow_dynamic_uri,
            "allowed_hosts": sorted(self.allowed_hosts) if self.allowed_hosts is not None else "any",
        }

    # ------------------------------------------------------------------ pools

    async def get_pool(self, uri: str) -> DbConnPool:
        """Return a connected pool for ``uri``, creating it if needed."""
        async with self._lock:
            self._evict_idle()
            entry = self._pools.get(uri)
            if entry is not None:
                entry.last_used = time.monotonic()
                self._pools.move_to_end(uri)
                return entry.pool
            key_lock = self._key_locks.setdefault(uri, asyncio.Lock())

        async with key_lock:
            # Someone else may have created it while we waited.
            async with self._lock:
                entry = self._pools.get(uri)
                if entry is not None:
                    entry.last_used = time.monotonic()
                    self._pools.move_to_end(uri)
                    return entry.pool

            await self._probe(uri)
            pool = DbConnPool(uri)
            await pool.pool_connect()  # raises ValueError with an obfuscated message on failure
            logger.info(f"Opened connection pool for {describe_conninfo(uri)}")

            async with self._lock:
                self._pools[uri] = _PoolEntry(pool=pool, pinned=(uri == self.default_uri))
                self._key_locks.pop(uri, None)
                self._evict_overflow()
            return pool

    async def _probe(self, uri: str) -> None:
        """Open one plain connection first: fails fast and with the real reason
        (bad password, unknown host, ...) instead of a generic pool timeout."""
        try:
            conn = await psycopg.AsyncConnection.connect(uri, connect_timeout=DEFAULT_CONNECT_TIMEOUT_SECONDS)
        except Exception as e:
            raise ValueError(f"Connection attempt failed for {describe_conninfo(uri)}: {obfuscate_password(str(e).strip())}") from e
        await conn.close()

    def _evict_idle(self) -> None:
        if self.idle_ttl <= 0:
            return
        now = time.monotonic()
        for key in [k for k, e in self._pools.items() if not e.pinned and now - e.last_used > self.idle_ttl]:
            self._evict(key, "idle")

    def _evict_overflow(self) -> None:
        while len(self._pools) > self.max_pools:
            victim = next((k for k, e in self._pools.items() if not e.pinned), None)
            if victim is None:
                return
            self._evict(victim, "cache full")

    def _evict(self, key: str, reason: str) -> None:
        entry = self._pools.pop(key)
        logger.info(f"Closing connection pool for {describe_conninfo(key)} ({reason})")
        # Close in the background: psycopg waits for in-flight connections to be returned.
        task = asyncio.create_task(entry.pool.close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    @property
    def pool_count(self) -> int:
        return len(self._pools)

    async def close_all(self) -> None:
        async with self._lock:
            entries = list(self._pools.values())
            self._pools.clear()
        await asyncio.gather(*(e.pool.close() for e in entries), *self._closing, return_exceptions=True)

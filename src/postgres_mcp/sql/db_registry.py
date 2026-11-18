"""Registry of the databases a server instance may work with, and their connection pools.

The set of databases is fixed by configuration, never by the model:

* environment variables ``<PREFIX><NAME>=<connection URI>`` (default prefix ``DATABASE_URI_``),
  e.g. ``DATABASE_URI_CRM=postgresql://...`` registers the database ``crm``;
* optionally an env file with the same ``KEY=VALUE`` lines, re-read whenever it changes,
  so databases can be added or removed without restarting the server;
* the legacy ``DATABASE_URI`` variable / command-line URI, registered as ``default``.

Tools refer to databases by name only. Connection strings coming from a tool call
or a request header are rejected.

Pools are created lazily, cached per database, and closed when idle for too long,
when the cache is full (least recently used first) or when the database disappears
from the configuration. The default database's pool is never evicted.
"""

import asyncio
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional

import psycopg
from psycopg.conninfo import conninfo_to_dict

from .sql_driver import DbConnPool
from .sql_driver import obfuscate_password

logger = logging.getLogger(__name__)

DEFAULT_ENV_PREFIX = "DATABASE_URI_"
DEFAULT_NAME = "default"
DEFAULT_MAX_POOLS = 10
DEFAULT_IDLE_TTL_SECONDS = 600.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class DatabaseResolutionError(ValueError):
    """Raised when the database for a request cannot be determined or is not allowed."""


def looks_like_conninfo(value: str) -> bool:
    """Whether a value is a connection string rather than a database name."""
    return "://" in value or "=" in value or "@" in value


def describe_conninfo(uri: str) -> str:
    """Human-readable description of a connection string without credentials."""
    try:
        params = conninfo_to_dict(uri)
    except Exception:
        return "<invalid connection string>"
    host = params.get("host") or "localhost"
    port = params.get("port") or "5432"
    dbname = params.get("dbname") or params.get("user") or ""
    return f"{host}:{port}/{dbname}"


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (``#`` comments, optional ``export`` and quotes)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key] = value
    return result


def databases_from_env(env: Mapping[str, str], prefix: str) -> dict[str, str]:
    """Collect ``<prefix><NAME>`` variables as ``{name: uri}`` (names lower-cased)."""
    return {
        key[len(prefix) :].lower(): value.strip() for key, value in env.items() if key.startswith(prefix) and len(key) > len(prefix) and value.strip()
    }


@dataclass
class _PoolEntry:
    pool: DbConnPool
    last_used: float = field(default_factory=time.monotonic)
    pinned: bool = False


class DatabaseRegistry:
    """Resolves which configured database a request targets and hands out cached pools."""

    def __init__(
        self,
        prefix: str = DEFAULT_ENV_PREFIX,
        env_file: Optional[str] = None,
        fallback_default_uri: Optional[str] = None,
        default_name: Optional[str] = None,
        max_pools: int = DEFAULT_MAX_POOLS,
        idle_ttl: float = DEFAULT_IDLE_TTL_SECONDS,
        environ: Optional[Mapping[str, str]] = None,
    ):
        self._pools: OrderedDict[str, _PoolEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._closing: set[asyncio.Task[None]] = set()
        self.configure(
            prefix=prefix,
            env_file=env_file,
            fallback_default_uri=fallback_default_uri,
            default_name=default_name,
            max_pools=max_pools,
            idle_ttl=idle_ttl,
            environ=environ,
        )

    def configure(
        self,
        prefix: str = DEFAULT_ENV_PREFIX,
        env_file: Optional[str] = None,
        fallback_default_uri: Optional[str] = None,
        default_name: Optional[str] = None,
        max_pools: int = DEFAULT_MAX_POOLS,
        idle_ttl: float = DEFAULT_IDLE_TTL_SECONDS,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Args:
            prefix: environment variable prefix that marks a database (``DATABASE_URI_`` -> ``DATABASE_URI_CRM``).
            env_file: optional file with extra ``KEY=VALUE`` lines, re-read when it changes.
            fallback_default_uri: legacy single database (``DATABASE_URI`` / CLI argument), registered as ``default``.
            default_name: database used when a call names none. Defaults to ``default`` if configured,
                or to the only database if exactly one is configured.
            environ: environment to read (``os.environ`` when omitted; read on every lookup).
        """
        if not prefix:
            raise ValueError("Database environment prefix must not be empty")
        self.prefix = prefix
        self.env_file = Path(env_file) if env_file else None
        self.fallback_default_uri = fallback_default_uri or None
        self.default_name = default_name.strip().lower() if default_name and default_name.strip() else None
        self.max_pools = max(1, max_pools)
        self.idle_ttl = idle_ttl
        self._environ = environ
        self._file_mtime: Optional[float] = None
        self._file_values: dict[str, str] = {}

    # ------------------------------------------------------------------ configuration

    def _read_env_file(self) -> dict[str, str]:
        if self.env_file is None:
            return {}
        try:
            mtime = self.env_file.stat().st_mtime
        except FileNotFoundError:
            if self._file_mtime is not None:
                logger.warning(f"Database env file {self.env_file} disappeared")
            self._file_mtime, self._file_values = None, {}
            return {}
        if mtime != self._file_mtime:
            try:
                self._file_values = parse_env_file(self.env_file.read_text())
                self._file_mtime = mtime
                logger.info(f"Loaded database env file {self.env_file}")
            except OSError as e:
                logger.warning(f"Could not read database env file {self.env_file}: {e}")
        return self._file_values

    def databases(self) -> dict[str, str]:
        """Current ``{name: uri}`` map. The env file overrides the process environment."""
        env = os.environ if self._environ is None else self._environ
        result: dict[str, str] = {}
        if self.fallback_default_uri:
            result[DEFAULT_NAME] = self.fallback_default_uri
        result.update(databases_from_env(env, self.prefix))
        result.update(databases_from_env(self._read_env_file(), self.prefix))
        return result

    def default_database(self, databases: Optional[dict[str, str]] = None) -> Optional[str]:
        """Name of the database used when a call names none, if any."""
        dbs = self.databases() if databases is None else databases
        if self.default_name:
            return self.default_name if self.default_name in dbs else None
        if DEFAULT_NAME in dbs:
            return DEFAULT_NAME
        if len(dbs) == 1:
            return next(iter(dbs))
        return None

    # ------------------------------------------------------------------ resolution

    def resolve(self, database: Optional[str] = None, header_value: Optional[str] = None) -> tuple[str, str]:
        """Pick ``(name, uri)`` for a request.

        ``database`` (tool argument) wins over ``header_value`` (HTTP header),
        which wins over the default database. Both must be configured names.
        """
        dbs = self.databases()
        for value, source in ((database, "tool argument 'database'"), (header_value, "X-Database header")):
            if isinstance(value, str) and value.strip():
                return self._resolve_name(value.strip(), source, dbs)

        default = self.default_database(dbs)
        if default is not None:
            return default, dbs[default]

        if not dbs:
            raise DatabaseResolutionError(f"No databases are configured on this server (set {self.prefix}<NAME> environment variables).")
        raise DatabaseResolutionError(
            f"No database specified and this server has no default database. Pass 'database', one of: {', '.join(sorted(dbs))}."
        )

    def _resolve_name(self, value: str, source: str, dbs: dict[str, str]) -> tuple[str, str]:
        known = ", ".join(sorted(dbs)) or "none"
        if looks_like_conninfo(value):
            # Never echo the value back: it may contain a password.
            raise DatabaseResolutionError(
                f"Connection strings are not accepted in the {source}. Use the name of a configured database (see list_databases): {known}."
            )
        name = value.lower()
        if name not in dbs:
            raise DatabaseResolutionError(f"Unknown database '{value}' ({source}). Available databases: {known}.")
        return name, dbs[name]

    def describe(self) -> list[dict[str, Any]]:
        """Configured databases, safe to show to a model (no credentials)."""
        dbs = self.databases()
        default = self.default_database(dbs)
        return [{"name": name, "target": describe_conninfo(uri), "default": name == default} for name, uri in sorted(dbs.items())]

    # ------------------------------------------------------------------ pools

    def _touch(self, uri: str) -> Optional[DbConnPool]:
        entry = self._pools.get(uri)
        if entry is None:
            return None
        entry.last_used = time.monotonic()
        self._pools.move_to_end(uri)
        return entry.pool

    async def get_pool(self, uri: str) -> DbConnPool:
        """Return a connected pool for ``uri`` (a value from ``resolve``), creating it if needed."""
        async with self._lock:
            self._evict_stale()
            pool = self._touch(uri)
            if pool is not None:
                return pool
            key_lock = self._key_locks.setdefault(uri, asyncio.Lock())

        async with key_lock:
            # Someone else may have created it while we waited.
            async with self._lock:
                pool = self._touch(uri)
                if pool is not None:
                    return pool

            await self._probe(uri)
            pool = DbConnPool(uri)
            await pool.pool_connect()  # raises ValueError with an obfuscated message on failure
            logger.info(f"Opened connection pool for {describe_conninfo(uri)}")

            async with self._lock:
                dbs = self.databases()
                default = self.default_database(dbs)
                self._pools[uri] = _PoolEntry(pool=pool, pinned=default is not None and dbs.get(default) == uri)
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

    def _evict_stale(self) -> None:
        """Close pools of databases removed from the configuration, and idle ones."""
        configured = set(self.databases().values())
        for key in [k for k in self._pools if k not in configured]:
            self._evict(key, "removed from configuration")
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

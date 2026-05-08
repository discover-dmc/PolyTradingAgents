"""Cache backend abstraction for PolyAgents.

Provides a uniform ``CacheBackend`` interface with two implementations:

- ``InMemoryCache`` — thread-safe in-process dict (default, zero deps)
- ``RedisCache``    — persistent Redis-backed cache (opt-in, requires ``redis``)

Switching backends requires only a config change:

    DEFAULT_CONFIG["cache_backend"] = "redis"
    DEFAULT_CONFIG["redis_url"]     = "redis://localhost:6379/0"

or at runtime:

    from polyagents.dataflows.cache import make_cache
    from polyagents.dataflows.interface import configure_session_cache
    configure_session_cache(make_cache("redis", namespace="pta:session"))
"""
from __future__ import annotations

import hashlib
import pickle
import threading
from typing import Any


# ---------------------------------------------------------------------------
# Base class / interface
# ---------------------------------------------------------------------------

class CacheBackend:
    """Minimal key-value store interface.

    Concrete implementations must override :meth:`get`, :meth:`set`, and
    :meth:`clear`.  Keys are always strings; values are arbitrary Python
    objects (implementations are responsible for serialisation).
    """

    def get(self, key: str) -> Any:
        """Return the stored value, or ``None`` if missing / expired."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key*.

        Args:
            key:   String cache key.
            value: Arbitrary Python object to cache.
            ttl:   Optional expiry in seconds.  ``None`` means no expiry.
                   Backends that do not support TTL must silently ignore it.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all entries owned by this backend instance."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------

class InMemoryCache(CacheBackend):
    """Thread-safe in-process dict cache.

    This is the default backend.  It requires no external dependencies and
    adds no infrastructure overhead.  Entries live until :meth:`clear` is
    called; ``ttl`` arguments are accepted for interface compatibility but
    silently ignored (in-memory entries never expire autonomously).
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ARG002
        with self._lock:
            self._store[key] = value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# RedisCache
# ---------------------------------------------------------------------------

class RedisCache(CacheBackend):
    """Redis-backed persistent cache.

    Requires the ``redis`` package and a reachable Redis server::

        pip install redis
        # or
        pip install "polyagents[redis]"

    All keys are namespaced under ``{namespace}:`` to avoid collisions with
    other applications sharing the same Redis instance.  Values are
    pickle-serialised so DataFrames and arbitrary Python objects are handled
    transparently.

    Unlike ``InMemoryCache``, this backend survives process restarts and is
    shared across multiple workers pointing at the same Redis instance.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        namespace: str = "pta",
    ) -> None:
        try:
            import redis as _redis
        except ImportError:
            raise ImportError(
                "The 'redis' package is required to use RedisCache.\n"
                "Install it with:  pip install redis\n"
                "or:               pip install 'polyagents[redis]'"
            ) from None
        self._redis = _redis.from_url(url, decode_responses=False)
        self._ns = namespace

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _k(self, key: str) -> str:
        """Prefix key with namespace."""
        return f"{self._ns}:{key}"

    # ------------------------------------------------------------------
    # CacheBackend interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        raw = self._redis.get(self._k(key))
        if raw is None:
            return None
        return pickle.loads(raw)  # noqa: S301

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if ttl is not None:
            self._redis.setex(self._k(key), ttl, raw)
        else:
            self._redis.set(self._k(key), raw)

    def clear(self) -> None:
        """Delete all keys in this namespace using non-blocking SCAN."""
        pattern = f"{self._ns}:*"
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match=pattern, count=200)
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_key(method: str, args: tuple) -> str:
    """Produce a stable fixed-length string key from ``(method, args)``.

    Uses pickle + SHA-256 so any picklable argument type (str, int, tuple,
    DataFrame, …) maps deterministically to a 64-char hex digest.
    """
    raw = pickle.dumps((method, args), protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(raw).hexdigest()


def make_cache(backend: str = "memory", **kwargs: Any) -> CacheBackend:
    """Factory: return a configured cache backend instance.

    Args:
        backend: ``"memory"`` (default) or ``"redis"``.
        **kwargs: Forwarded to the backend constructor.
            Relevant for Redis: ``url`` (connection string) and
            ``namespace`` (key prefix, default ``"pta"``).
            Unknown kwargs are silently ignored by ``InMemoryCache``.

    Example::

        # Default — in-memory, no extra deps
        cache = make_cache()

        # Redis with custom URL and namespace
        cache = make_cache("redis", url="redis://myhost:6379/1",
                           namespace="pta:session")
    """
    if backend == "redis":
        return RedisCache(
            url=kwargs.get("url", "redis://localhost:6379/0"),
            namespace=kwargs.get("namespace", "pta"),
        )
    return InMemoryCache()

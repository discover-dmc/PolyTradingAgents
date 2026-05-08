"""Unit tests for the CacheBackend abstraction.

Covers InMemoryCache behaviour, the make_key helper, and the make_cache
factory.  RedisCache is tested only for the ImportError path — we don't
require a live Redis instance in the test suite.
"""
import builtins
import unittest.mock as mock

import pytest

from polyagents.dataflows.cache import (
    CacheBackend,
    InMemoryCache,
    make_cache,
    make_key,
)


@pytest.mark.unit
class TestInMemoryCache:
    def test_get_miss_returns_none(self):
        c = InMemoryCache()
        assert c.get("missing") is None

    def test_set_get_roundtrip(self):
        c = InMemoryCache()
        c.set("k", "hello")
        assert c.get("k") == "hello"

    def test_set_complex_value(self):
        import pandas as pd
        c = InMemoryCache()
        df = pd.DataFrame({"a": [1, 2, 3]})
        c.set("df", df)
        result = c.get("df")
        assert list(result["a"]) == [1, 2, 3]

    def test_set_with_ttl_does_not_error(self):
        """InMemoryCache silently ignores the ttl argument."""
        c = InMemoryCache()
        c.set("k", 42, ttl=60)
        assert c.get("k") == 42

    def test_overwrite(self):
        c = InMemoryCache()
        c.set("k", "first")
        c.set("k", "second")
        assert c.get("k") == "second"

    def test_clear_removes_all_entries(self):
        c = InMemoryCache()
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.get("a") is None
        assert c.get("b") is None

    def test_clear_on_empty_is_safe(self):
        c = InMemoryCache()
        c.clear()  # should not raise

    def test_instances_are_independent(self):
        c1, c2 = InMemoryCache(), InMemoryCache()
        c1.set("x", "from-c1")
        assert c2.get("x") is None

    def test_is_cache_backend_subclass(self):
        assert isinstance(InMemoryCache(), CacheBackend)


@pytest.mark.unit
class TestMakeKey:
    def test_same_args_produce_same_key(self):
        assert make_key("get_news", ("query", 7)) == make_key("get_news", ("query", 7))

    def test_different_methods_produce_different_keys(self):
        assert make_key("get_news", ("x",)) != make_key("get_data", ("x",))

    def test_different_args_produce_different_keys(self):
        assert make_key("m", ("a",)) != make_key("m", ("b",))

    def test_returns_hex_string(self):
        k = make_key("method", ("arg",))
        assert isinstance(k, str)
        assert len(k) == 64          # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in k)

    def test_empty_args(self):
        k = make_key("method", ())
        assert isinstance(k, str) and len(k) == 64

    def test_numeric_args(self):
        assert make_key("m", (1, 2)) != make_key("m", (1, 3))


@pytest.mark.unit
class TestMakeCache:
    def test_default_returns_in_memory(self):
        c = make_cache()
        assert isinstance(c, InMemoryCache)

    def test_explicit_memory_returns_in_memory(self):
        c = make_cache("memory")
        assert isinstance(c, InMemoryCache)

    def test_memory_cache_ignores_redis_kwargs(self):
        """Extra kwargs must not cause an error for the memory backend."""
        c = make_cache("memory", url="redis://irrelevant", namespace="ns")
        assert isinstance(c, InMemoryCache)

    def test_redis_raises_import_error_when_package_missing(self):
        real_import = builtins.__import__

        def _block_redis(name, *args, **kwargs):
            if name == "redis":
                raise ImportError("No module named 'redis'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_block_redis):
            with pytest.raises(ImportError, match="redis"):
                make_cache("redis")

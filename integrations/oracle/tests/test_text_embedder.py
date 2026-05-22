# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json

import oracledb
import pytest

from haystack_integrations.components.embedders.oracle import OracleTextEmbedder
from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connect_dsn,
    oracle_unit_test_connection_params,
)

default_params = {
    "connection_params": oracle_unit_test_connection_params(),
    "embedding_params": {"provider": "database", "model": "ALL_MINILM_L12_V2"},
    "proxy": None,
    "use_connection_pool": False,
}


class _FakeCursor:
    def __init__(self):
        self.rows = [(json.dumps({"embed_vector": json.dumps([0.1, 0.2])}),)]
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, params=None, **kwargs):
        self.query = query
        self.executed.append((query, params, kwargs))

    def setinputsizes(self, *args, **kwargs):
        return None

    def __iter__(self):
        return iter(self.rows)


class _FakeVectorArrayType:
    def newobject(self, values):
        return values


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def gettype(self, name):
        return _FakeVectorArrayType()


class _FakePooledConnection:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        self.released["value"] = True
        return None


class _FakePool:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    def acquire(self):
        return _FakePooledConnection(self.connection, self.released)


class _AsyncFakeCursor:
    def __init__(self):
        self.rows = [(json.dumps({"embed_vector": json.dumps([0.1, 0.2])}),)]
        self.executed = []

    async def execute(self, query, params=None, **kwargs):
        self.query = query
        self.executed.append((query, params, kwargs))

    def setinputsizes(self, *args, **kwargs):
        return None

    async def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _AsyncFakeVectorArrayType:
    def newobject(self):
        return []


class _AsyncFakeClob:
    def __init__(self):
        self.value = ""

    async def write(self, value):
        self.value = value


class _AsyncFakeConnection:
    def __init__(self):
        self.cursor_obj = _AsyncFakeCursor()

    def cursor(self):
        return self.cursor_obj

    async def gettype(self, name):
        return _AsyncFakeVectorArrayType()

    async def createlob(self, db_type):
        return _AsyncFakeClob()


class _AsyncFakePooledConnection:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        self.released["value"] = True
        return None


class _AsyncFakePool:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    def acquire(self):
        return _AsyncFakePooledConnection(self.connection, self.released)


class TestOracleTextEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for OracleTextEmbedder.
        """
        OracleTextEmbedder(**default_params)

    def test_to_dict(self):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        embedder_component = OracleTextEmbedder(**default_params)
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder",
            "init_parameters": {**default_params},
        }

    def test_from_dict(self):
        component_dict = {
            "type": "haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder",
            "init_parameters": {**default_params},
        }

        OracleTextEmbedder.from_dict(component_dict)

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embedding.
        """
        embedder = OracleTextEmbedder(**default_params)
        list_integers_input = ["text_snippet_1", "text_snippet_2"]

        with pytest.raises(TypeError):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
    @pytest.mark.integration
    def test_run(self):
        embedder = OracleTextEmbedder(
            **{
                **default_params,
                "connection_params": {"dsn": oracle_test_connect_dsn()},
            }
        )
        text = "The food was delicious"
        result = embedder.run(text=text)

        assert len(result["embedding"]) > 0
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
    @pytest.mark.integration
    async def test_run_async(self):
        embedder = OracleTextEmbedder(
            **{
                **default_params,
                "connection_params": {"dsn": oracle_test_connect_dsn()},
            }
        )
        text = "The food was delicious"
        result = await embedder.run_async(text=text)

        assert len(result["embedding"]) > 0
        assert all(isinstance(x, float) for x in result["embedding"])


def test_embed_documents_releases_sync_pooled_connection(monkeypatch):
    released = {"value": False}
    pool = _FakePool(_FakeConnection(), released)

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True})
    embedder._client = pool
    embedder._initialized = True

    assert embedder._embed_documents(["hello"]) == [[0.1, 0.2]]
    assert released["value"] is True


def test_ensure_initialized_uses_pool_and_old_versions_fail(monkeypatch):
    created = {}

    def fake_create_pool(**kwargs):
        created["pool"] = kwargs
        return object()

    monkeypatch.setattr(oracledb, "create_pool", fake_create_pool)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True})
    embedder._ensure_initialized()
    assert created["pool"] == default_params["connection_params"]

    embedder = OracleTextEmbedder(**default_params)
    monkeypatch.setattr(oracledb, "__version__", "2.1.0")
    with pytest.raises(Exception, match="must be >=2.2.0"):
        embedder._ensure_initialized()


def test_ensure_initialized_uses_direct_connect_and_run(monkeypatch):
    created = {}

    def fake_connect(**kwargs):
        created["connect"] = kwargs
        return object()

    monkeypatch.setattr(oracledb, "connect", fake_connect)
    embedder = OracleTextEmbedder(**default_params)
    embedder._ensure_initialized()
    assert created["connect"] == default_params["connection_params"]
    assert embedder._client is not None

    monkeypatch.setattr(embedder, "_embed_documents", lambda texts: [[0.3, 0.4]] if texts == ["hello"] else [[]])
    assert embedder.run("hello") == {"embedding": [0.3, 0.4], "meta": default_params["embedding_params"]}


def test_embed_documents_sync_proxy_and_empty_row(monkeypatch):
    released = {"value": False}
    connection = _FakeConnection()
    connection.cursor_obj.rows = [None]
    pool = _FakePool(connection, released)

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client = pool
    embedder._initialized = True

    assert embedder._embed_documents(["hello"]) == [[]]
    assert connection.cursor_obj.executed[0][0] == "begin utl_http.set_proxy(:proxy); end;"


@pytest.mark.asyncio
async def test_embed_documents_async_releases_pooled_connection(monkeypatch):
    released = {"value": False}
    pool = _AsyncFakePool(_AsyncFakeConnection(), released)

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True})
    embedder._client_async = pool
    embedder._initialized_async = True

    assert await embedder._embed_documents_async(["hello"]) == [[0.1, 0.2]]
    assert released["value"] is True


@pytest.mark.asyncio
async def test_ensure_initialized_async_uses_pool_and_old_versions_fail(monkeypatch):
    created = {}

    async def fake_create_pool_async(**kwargs):
        created["pool"] = kwargs
        return object()

    monkeypatch.setattr(oracledb, "create_pool_async", fake_create_pool_async)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True})
    await embedder._ensure_initialized_async()
    assert created["pool"] == default_params["connection_params"]
    assert embedder._client_async is not None

    embedder = OracleTextEmbedder(**default_params)
    monkeypatch.setattr(oracledb, "__version__", "2.1.0")
    with pytest.raises(Exception, match="must be >=2.2.0"):
        await embedder._ensure_initialized_async()


@pytest.mark.asyncio
async def test_ensure_initialized_async_uses_direct_connect_and_run_async(monkeypatch):
    created = {}

    async def fake_connect_async(**kwargs):
        created["connect"] = kwargs
        return object()

    monkeypatch.setattr(oracledb, "connect_async", fake_connect_async)
    embedder = OracleTextEmbedder(**default_params)
    await embedder._ensure_initialized_async()
    assert created["connect"] == default_params["connection_params"]
    assert embedder._client_async is not None

    async def fake_embed_documents_async(texts):
        assert texts == ["hello"]
        return [[0.3, 0.4]]

    monkeypatch.setattr(embedder, "_embed_documents_async", fake_embed_documents_async)
    assert await embedder.run_async("hello") == {
        "embedding": [0.3, 0.4],
        "meta": default_params["embedding_params"],
    }


@pytest.mark.asyncio
async def test_embed_documents_async_proxy_and_empty_row(monkeypatch):
    released = {"value": False}
    connection = _AsyncFakeConnection()
    connection.cursor_obj.rows = [None]
    pool = _AsyncFakePool(connection, released)

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client_async = pool
    embedder._initialized_async = True

    assert await embedder._embed_documents_async(["hello"]) == [[]]
    assert connection.cursor_obj.executed[0][0] == "begin utl_http.set_proxy(:proxy); end;"


@pytest.mark.asyncio
async def test_run_async_wrong_input_format():
    embedder = OracleTextEmbedder(**default_params)

    with pytest.raises(TypeError):
        await embedder.run_async(text=["text_snippet_1"])  # type: ignore[arg-type]

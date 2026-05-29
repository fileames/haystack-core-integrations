# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json

import oracledb
import pytest
from haystack.utils import Secret

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
        self.raise_on_embedding = None
        self.raise_on_clear_proxy = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, params=None, **kwargs):
        self.query = query
        self.executed.append((query, params, kwargs))
        if self.raise_on_clear_proxy and "utl_http.set_proxy" in query and kwargs.get("proxy") is None:
            raise self.raise_on_clear_proxy
        if self.raise_on_embedding and "utl_to_embeddings" in query:
            raise self.raise_on_embedding

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
        self.raise_on_embedding = None
        self.raise_on_clear_proxy = None

    async def execute(self, query, params=None, **kwargs):
        self.query = query
        self.executed.append((query, params, kwargs))
        if self.raise_on_clear_proxy and "utl_http.set_proxy" in query and kwargs.get("proxy") is None:
            raise self.raise_on_clear_proxy
        if self.raise_on_embedding and "utl_to_embeddings" in query:
            raise self.raise_on_embedding

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
            "init_parameters": {**default_params, "connection_params": {"user": None, "password": None, "dsn": None}},
        }

    def test_to_dict_serializes_secret_connection_params_and_proxy(self):
        embedder_component = OracleTextEmbedder(
            **{
                **default_params,
                "connection_params": {
                    "user": Secret.from_env_var("ORACLE_USER"),
                    "password": Secret.from_env_var("ORACLE_PASSWORD"),
                    "dsn": Secret.from_env_var("ORACLE_DSN"),
                },
                "proxy": Secret.from_env_var("ORACLE_PROXY"),
            }
        )

        component_dict = embedder_component.to_dict()
        assert component_dict["init_parameters"]["connection_params"] == {
            "user": {"type": "env_var", "env_vars": ["ORACLE_USER"], "strict": True},
            "password": {"type": "env_var", "env_vars": ["ORACLE_PASSWORD"], "strict": True},
            "dsn": {"type": "env_var", "env_vars": ["ORACLE_DSN"], "strict": True},
        }
        assert component_dict["init_parameters"]["proxy"] == {
            "type": "env_var",
            "env_vars": ["ORACLE_PROXY"],
            "strict": True,
        }

    def test_to_dict_omits_plain_sensitive_connection_params_and_proxy(self):
        embedder_component = OracleTextEmbedder(
            **{
                **default_params,
                "connection_params": {
                    "user": "onnxuser",
                    "password": "secret",
                    "dsn": "onnxuser/secret@database.example/pdb",
                    "events": True,
                },
                "proxy": "http://proxy_user:proxy_password@proxy.example:80",
            }
        )

        component_dict = embedder_component.to_dict()
        assert component_dict["init_parameters"]["connection_params"] == {
            "user": None,
            "password": None,
            "dsn": None,
            "events": True,
        }
        assert component_dict["init_parameters"]["proxy"] is None

    def test_from_dict(self):
        component_dict = {
            "type": "haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder",
            "init_parameters": {**default_params},
        }

        OracleTextEmbedder.from_dict(component_dict)

    def test_from_dict_deserializes_secret_connection_params_and_proxy(self, monkeypatch):
        monkeypatch.setenv("ORACLE_USER", "onnxuser")
        monkeypatch.setenv("ORACLE_PASSWORD", "secret")
        monkeypatch.setenv("ORACLE_DSN", "database.example/pdb")
        monkeypatch.setenv("ORACLE_PROXY", "http://proxy")

        embedder = OracleTextEmbedder.from_dict(
            {
                "type": "haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder",
                "init_parameters": {
                    **default_params,
                    "connection_params": {
                        "user": {"type": "env_var", "env_vars": ["ORACLE_USER"], "strict": True},
                        "password": {"type": "env_var", "env_vars": ["ORACLE_PASSWORD"], "strict": True},
                        "dsn": {"type": "env_var", "env_vars": ["ORACLE_DSN"], "strict": True},
                    },
                    "proxy": {"type": "env_var", "env_vars": ["ORACLE_PROXY"], "strict": True},
                },
            }
        )

        assert embedder._connection_params["user"].resolve_value() == "onnxuser"
        assert embedder._connection_params["password"].resolve_value() == "secret"
        assert embedder._connection_params["dsn"].resolve_value() == "database.example/pdb"
        assert embedder._proxy.resolve_value() == "http://proxy"

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


def test_ensure_initialized_resolves_secret_connection_params(monkeypatch):
    monkeypatch.setenv("ORACLE_USER", "onnxuser")
    monkeypatch.setenv("ORACLE_PASSWORD", "secret")
    monkeypatch.setenv("ORACLE_DSN", "database.example/pdb")
    created = {}

    def fake_connect(**kwargs):
        created["connect"] = kwargs
        return object()

    monkeypatch.setattr(oracledb, "connect", fake_connect)
    embedder = OracleTextEmbedder(
        **{
            **default_params,
            "connection_params": {
                "user": Secret.from_env_var("ORACLE_USER"),
                "password": Secret.from_env_var("ORACLE_PASSWORD"),
                "dsn": Secret.from_env_var("ORACLE_DSN"),
            },
        }
    )
    embedder._ensure_initialized()

    assert created["connect"] == {
        "user": "onnxuser",
        "password": "secret",
        "dsn": "database.example/pdb",
    }


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
    assert [
        kwargs.get("proxy")
        for query, _params, kwargs in connection.cursor_obj.executed
        if query == "begin utl_http.set_proxy(:proxy); end;"
    ] == ["http://proxy", None]


def test_embed_documents_sync_proxy_cleared_on_exception(monkeypatch):
    released = {"value": False}
    connection = _FakeConnection()
    connection.cursor_obj.raise_on_embedding = RuntimeError("embedding failed")
    pool = _FakePool(connection, released)

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client = pool
    embedder._initialized = True

    with pytest.raises(RuntimeError, match="embedding failed"):
        embedder._embed_documents(["hello"])

    assert [
        kwargs.get("proxy")
        for query, _params, kwargs in connection.cursor_obj.executed
        if query == "begin utl_http.set_proxy(:proxy); end;"
    ] == ["http://proxy", None]
    assert released["value"] is True


def test_embed_documents_sync_proxy_cleanup_failure_after_embedding_failure_raises(monkeypatch):
    released = {"value": False}
    connection = _FakeConnection()
    connection.cursor_obj.raise_on_embedding = RuntimeError("embedding failed")
    connection.cursor_obj.raise_on_clear_proxy = RuntimeError("clear failed")
    pool = _FakePool(connection, released)

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client = pool
    embedder._initialized = True

    with pytest.raises(RuntimeError, match="Failed to clear Oracle session proxy after embedding failed"):
        embedder._embed_documents(["hello"])

    assert [
        kwargs.get("proxy")
        for query, _params, kwargs in connection.cursor_obj.executed
        if query == "begin utl_http.set_proxy(:proxy); end;"
    ] == ["http://proxy", None]
    assert released["value"] is True


def test_embed_documents_sync_proxy_cleanup_failure_raises(monkeypatch):
    released = {"value": False}
    connection = _FakeConnection()
    connection.cursor_obj.raise_on_clear_proxy = RuntimeError("clear failed")
    pool = _FakePool(connection, released)

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client = pool
    embedder._initialized = True

    with pytest.raises(RuntimeError, match="Failed to clear Oracle session proxy after embedding succeeded"):
        embedder._embed_documents(["hello"])

    assert released["value"] is True


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
    assert [
        kwargs.get("proxy")
        for query, _params, kwargs in connection.cursor_obj.executed
        if query == "begin utl_http.set_proxy(:proxy); end;"
    ] == ["http://proxy", None]


@pytest.mark.asyncio
async def test_embed_documents_async_proxy_cleared_on_exception(monkeypatch):
    released = {"value": False}
    connection = _AsyncFakeConnection()
    connection.cursor_obj.raise_on_embedding = RuntimeError("embedding failed")
    pool = _AsyncFakePool(connection, released)

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client_async = pool
    embedder._initialized_async = True

    with pytest.raises(RuntimeError, match="embedding failed"):
        await embedder._embed_documents_async(["hello"])

    assert [
        kwargs.get("proxy")
        for query, _params, kwargs in connection.cursor_obj.executed
        if query == "begin utl_http.set_proxy(:proxy); end;"
    ] == ["http://proxy", None]
    assert released["value"] is True


@pytest.mark.asyncio
async def test_embed_documents_async_proxy_cleanup_failure_after_embedding_failure_raises(monkeypatch):
    released = {"value": False}
    connection = _AsyncFakeConnection()
    connection.cursor_obj.raise_on_embedding = RuntimeError("embedding failed")
    connection.cursor_obj.raise_on_clear_proxy = RuntimeError("clear failed")
    pool = _AsyncFakePool(connection, released)

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client_async = pool
    embedder._initialized_async = True

    with pytest.raises(RuntimeError, match="Failed to clear Oracle session proxy after embedding failed"):
        await embedder._embed_documents_async(["hello"])

    assert [
        kwargs.get("proxy")
        for query, _params, kwargs in connection.cursor_obj.executed
        if query == "begin utl_http.set_proxy(:proxy); end;"
    ] == ["http://proxy", None]
    assert released["value"] is True


@pytest.mark.asyncio
async def test_embed_documents_async_proxy_cleanup_failure_raises(monkeypatch):
    released = {"value": False}
    connection = _AsyncFakeConnection()
    connection.cursor_obj.raise_on_clear_proxy = RuntimeError("clear failed")
    pool = _AsyncFakePool(connection, released)

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    embedder = OracleTextEmbedder(**{**default_params, "use_connection_pool": True, "proxy": "http://proxy"})
    embedder._client_async = pool
    embedder._initialized_async = True

    with pytest.raises(RuntimeError, match="Failed to clear Oracle session proxy after embedding succeeded"):
        await embedder._embed_documents_async(["hello"])

    assert released["value"] is True


@pytest.mark.asyncio
async def test_run_async_wrong_input_format():
    embedder = OracleTextEmbedder(**default_params)

    with pytest.raises(TypeError):
        await embedder.run_async(text=["text_snippet_1"])  # type: ignore[arg-type]

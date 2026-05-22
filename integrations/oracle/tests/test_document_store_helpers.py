# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager, contextmanager
import json
from types import SimpleNamespace

import pytest
from haystack.dataclasses import SparseEmbedding
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.errors import FilterError

from haystack_integrations.components.document_stores.oracle import document_store as ds
from haystack_integrations.components.embedders.oracle import OracleTextEmbedder
from haystack_integrations.components.document_stores.oracle.filters import (
    _convert_oper_to_sql,
    _get_filter_string,
    _to_hybrid_filter,
)

from .conftest import oracle_unit_test_connection_params


class _ErrorInfo:
    def __init__(self, code: int):
        self.code = code


class _FakeIntegrityError(Exception):
    pass


class _FakeOracleError(Exception):
    pass


class _FakeDatabaseError(Exception):
    pass


class _DirectConnection:
    pass


class _AsyncDirectConnection:
    pass


class _PoolContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return None


class _AsyncPoolContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _PoolContext(self.connection)


class _AsyncPool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _AsyncPoolContext(self.connection)


class _RaisingCursor:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, *args, **kwargs):
        raise self.exc


class _AsyncRaisingCursor:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    async def execute(self, *args, **kwargs):
        raise self.exc


class _ExecCursor:
    def __init__(self, fetchone_result=None, fetchall_result=None, description=None):
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.description = description or []
        self.executed = []
        self.executemany_calls = []
        self.outputtypehandler = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, *args, **kwargs):
        self.executed.append((query, args, kwargs))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def setinputsizes(self, *args, **kwargs):
        return None

    def executemany(self, query, params):
        self.executemany_calls.append((query, params))
        self.rowcount = len(params)


class _AsyncExecCursor(_ExecCursor):
    async def execute(self, query, *args, **kwargs):
        self.executed.append((query, args, kwargs))

    async def fetchone(self):
        return self.fetchone_result

    async def fetchall(self):
        return self.fetchall_result

    async def executemany(self, query, params):
        self.executemany_calls.append((query, params))
        self.rowcount = len(params)


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


class _AsyncConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    async def commit(self):
        self.committed = True


class _SparseValue:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


def test_handle_exceptions_duplicate_document_error(monkeypatch):
    monkeypatch.setattr(ds.oracledb, "IntegrityError", _FakeIntegrityError)

    @ds._handle_exceptions
    def boom():
        raise _FakeIntegrityError(_ErrorInfo(1))

    with pytest.raises(DuplicateDocumentError):
        boom()


def test_handle_exceptions_runtime_value_filter_and_generic(monkeypatch):
    monkeypatch.setattr(ds.oracledb, "IntegrityError", _FakeIntegrityError)
    monkeypatch.setattr(ds.oracledb, "Error", _FakeOracleError)

    @ds._handle_exceptions
    def non_dup_integrity():
        raise _FakeIntegrityError(_ErrorInfo(2))

    @ds._handle_exceptions
    def db_err():
        raise _FakeOracleError("db")

    @ds._handle_exceptions
    def runtime_err():
        raise RuntimeError("runtime")

    @ds._handle_exceptions
    def value_err():
        raise ValueError("value")

    @ds._handle_exceptions
    def filter_err():
        raise FilterError("filter")

    @ds._handle_exceptions
    def generic_err():
        raise Exception("boom")

    with pytest.raises(RuntimeError, match="Failed due to a DB error"):
        non_dup_integrity()
    with pytest.raises(RuntimeError, match="Failed due to a DB error"):
        db_err()
    with pytest.raises(RuntimeError, match="Failed due to a runtime error"):
        runtime_err()
    with pytest.raises(ValueError, match="Validation failed"):
        value_err()
    with pytest.raises(FilterError, match="filter"):
        filter_err()
    with pytest.raises(RuntimeError, match="Unexpected error"):
        generic_err()


@pytest.mark.asyncio
async def test_handle_exceptions_async_branches(monkeypatch):
    monkeypatch.setattr(ds.oracledb, "IntegrityError", _FakeIntegrityError)
    monkeypatch.setattr(ds.oracledb, "Error", _FakeOracleError)

    @ds._handle_exceptions_async
    async def dup():
        raise _FakeIntegrityError(_ErrorInfo(1))

    @ds._handle_exceptions_async
    async def non_dup():
        raise _FakeIntegrityError(_ErrorInfo(99))

    @ds._handle_exceptions_async
    async def db():
        raise _FakeOracleError("db")

    @ds._handle_exceptions_async
    async def runtime():
        raise RuntimeError("runtime")

    @ds._handle_exceptions_async
    async def value():
        raise ValueError("value")

    @ds._handle_exceptions_async
    async def filter_error():
        raise FilterError("filter")

    @ds._handle_exceptions_async
    async def generic():
        raise Exception("boom")

    with pytest.raises(DuplicateDocumentError):
        await dup()
    with pytest.raises(RuntimeError, match="Failed due to a DB error"):
        await non_dup()
    with pytest.raises(RuntimeError, match="Failed due to a DB error"):
        await db()
    with pytest.raises(RuntimeError, match="Failed due to a runtime error"):
        await runtime()
    with pytest.raises(ValueError, match="Validation failed"):
        await value()
    with pytest.raises(FilterError, match="filter"):
        await filter_error()
    with pytest.raises(RuntimeError, match="Unexpected error"):
        await generic()


def test_get_connection_variants(monkeypatch):
    monkeypatch.setattr(ds.oracledb, "Connection", _DirectConnection)
    monkeypatch.setattr(ds.oracledb, "ConnectionPool", _Pool)

    direct = _DirectConnection()
    with ds._get_connection(direct) as connection:
        assert connection is direct

    pooled_connection = _DirectConnection()
    with ds._get_connection(_Pool(pooled_connection)) as connection:
        assert connection is pooled_connection

    with pytest.raises(TypeError, match="Expected client of type"):
        with ds._get_connection(object()):
            pass

    monkeypatch.delattr(ds.oracledb, "ConnectionPool", raising=False)
    with pytest.raises(TypeError, match="Expected client of type oracledb.Connection"):
        with ds._get_connection(object()):
            pass


@pytest.mark.asyncio
async def test_get_connection_async_variants(monkeypatch):
    monkeypatch.setattr(ds.oracledb, "AsyncConnection", _AsyncDirectConnection)
    monkeypatch.setattr(ds.oracledb, "AsyncConnectionPool", _AsyncPool)

    direct = _AsyncDirectConnection()
    async with ds._get_connection_async(direct) as connection:
        assert connection is direct

    pooled_connection = _AsyncDirectConnection()
    async with ds._get_connection_async(_AsyncPool(pooled_connection)) as connection:
        assert connection is pooled_connection

    with pytest.raises(TypeError, match="Expected client of type"):
        async with ds._get_connection_async(object()):
            pass

    monkeypatch.delattr(ds.oracledb, "AsyncConnectionPool", raising=False)
    with pytest.raises(TypeError, match="Expected client of type oracledb.AsyncConnection"):
        async with ds._get_connection_async(object()):
            pass


def test_compare_version_and_table_exists(monkeypatch):
    assert ds._compare_version("2.1.0", "2.2.0") is True
    assert ds._compare_version("2.3.0", "2.2.0") is False
    assert ds._compare_version("2.2", "2.2.0") is True

    monkeypatch.setattr(ds.oracledb, "DatabaseError", _FakeDatabaseError)

    true_conn = _Connection(_ExecCursor())
    assert ds._table_exists(true_conn, "docs") is True

    false_conn = _Connection(_RaisingCursor(_FakeDatabaseError(_ErrorInfo(ds.DUPLICATE_ERROR))))
    assert ds._table_exists(false_conn, "docs") is False

    bad_conn = _Connection(_RaisingCursor(_FakeDatabaseError(_ErrorInfo(999))))
    with pytest.raises(_FakeDatabaseError):
        ds._table_exists(bad_conn, "docs")


@pytest.mark.asyncio
async def test_table_exists_async_false_and_reraise(monkeypatch):
    monkeypatch.setattr(ds.oracledb, "DatabaseError", _FakeDatabaseError)

    true_conn = _AsyncConnection(_AsyncExecCursor())
    assert await ds._table_exists_async(true_conn, "docs") is True

    false_conn = _AsyncConnection(_AsyncRaisingCursor(_FakeDatabaseError(_ErrorInfo(ds.DUPLICATE_ERROR))))
    assert await ds._table_exists_async(false_conn, "docs") is False

    bad_conn = _AsyncConnection(_AsyncRaisingCursor(_FakeDatabaseError(_ErrorInfo(999))))
    with pytest.raises(_FakeDatabaseError):
        await ds._table_exists_async(bad_conn, "docs")


def test_identifier_and_index_query_helpers():
    assert ds._quote_identifier("docs") == '"docs"'
    assert ds._quote_identifier('schema."docs"') == '"schema"."docs"'
    with pytest.raises(ValueError, match="not valid"):
        ds._quote_identifier('schema."docs')

    query, params = ds._get_index_exists_query('"IDX"', '"TAB"')
    assert "table_name" in query
    assert params == {"idx_name": "IDX", "table_name": "TAB"}

    query, params = ds._get_index_exists_query('"IDX"', None)
    assert "table_name" not in params
    assert params == {"idx_name": "IDX"}

    idx_name = ds._get_index_name("HNSW")
    assert idx_name.startswith('"HNSW_')
    assert idx_name.endswith('"')


def test_index_exists_and_output_handler():
    cursor = _ExecCursor(fetchone_result=("IDX",))
    assert ds._index_exists(_Connection(cursor), '"IDX"', '"TAB"') is True

    cursor = _ExecCursor(fetchone_result=None)
    assert ds._index_exists(_Connection(cursor), '"IDX"') is False


def test_text_index_helpers():
    assert ds._validate_text_index_column("content") == "content"
    assert ds._get_text_index_ddl('"DOCS"', '"IDX_TEXT"') == 'CREATE SEARCH INDEX "IDX_TEXT" ON "DOCS"(content)'

    with pytest.raises(ValueError, match="supports only the 'content' column"):
        ds._validate_text_index_column("blob_mime_type")


def test_create_text_index_executes_when_missing(monkeypatch):
    cursor = _ExecCursor()
    connection = _Connection(cursor)

    monkeypatch.setattr(ds, "_index_exists", lambda *_args, **_kwargs: False)

    ds._create_text_index(connection, '"DOCS"', '"IDX_TEXT"')

    assert cursor.executed == [('CREATE SEARCH INDEX "IDX_TEXT" ON "DOCS"(content)', (), {})]


@pytest.mark.asyncio
async def test_create_text_index_async_executes_when_missing(monkeypatch):
    cursor = _AsyncExecCursor()
    connection = _AsyncConnection(cursor)

    async def index_missing(*_args, **_kwargs):
        return False

    monkeypatch.setattr(ds, "_index_exists_async", index_missing)

    await ds._create_text_index_async(connection, '"DOCS"', '"IDX_TEXT"')

    assert cursor.executed == [('CREATE SEARCH INDEX "IDX_TEXT" ON "DOCS"(content)', (), {})]

    class _VarCursor:
        arraysize = 10

        def var(self, db_type, arraysize=None):
            return db_type, arraysize

    metadata = SimpleNamespace(type_code=ds.oracledb.DB_TYPE_CLOB)
    assert ds.output_type_string_handler(_VarCursor(), metadata)[0] == ds.oracledb.DB_TYPE_LONG
    metadata.type_code = ds.oracledb.DB_TYPE_NCLOB
    assert ds.output_type_string_handler(_VarCursor(), metadata)[0] == ds.oracledb.DB_TYPE_LONG_NVARCHAR
    metadata.type_code = ds.oracledb.DB_TYPE_BLOB
    assert ds.output_type_string_handler(_VarCursor(), metadata)[0] == ds.oracledb.DB_TYPE_LONG_RAW
    metadata.type_code = object()
    assert ds.output_type_string_handler(_VarCursor(), metadata) is None


@pytest.mark.asyncio
async def test_index_exists_async(monkeypatch):
    cursor = _AsyncExecCursor(fetchone_result=("IDX",))
    assert await ds._index_exists_async(_AsyncConnection(cursor), '"IDX"', '"TAB"') is True

    cursor = _AsyncExecCursor(fetchone_result=None)
    assert await ds._index_exists_async(_AsyncConnection(cursor), '"IDX"') is False


def test_index_ddl_builders():
    idx_name, ddl = ds._get_hnsw_index_ddl("docs", "cosine")
    assert idx_name.startswith('"HNSW_')
    assert "DISTANCE cosine" in ddl
    assert "efconstruction 200" in ddl

    _, ddl = ds._get_hnsw_index_ddl("docs", "dot", {"idx_type": "HNSW", "neighbors": 10})
    assert "neighbors 10" in ddl
    assert "efconstruction 200" in ddl

    _, ddl = ds._get_hnsw_index_ddl("docs", "dot", {"idx_type": "HNSW", "efconstruction": 10})
    assert "neighbors 32" in ddl
    assert "efconstruction 10" in ddl

    with pytest.raises(ValueError, match="Invalid parameter"):
        ds._get_hnsw_index_ddl("docs", "dot", {"idx_type": "HNSW", "bogus": 1})

    with pytest.raises(ValueError, match="Invalid parameter"):
        ds._get_hnsw_index_ddl("docs", "dot", {"idx_type": "HNSW", "efConstruction": 10})

    with pytest.raises(ValueError, match="parallel must be an integer"):
        ds._get_hnsw_index_ddl("docs", "dot", {"idx_type": "HNSW", "parallel": "8 NOLOGGING"})

    with pytest.raises(ValueError, match="accuracy must be at most 100"):
        ds._get_hnsw_index_ddl("docs", "dot", {"idx_type": "HNSW", "accuracy": 101})

    idx_name, ddl = ds._get_ivf_index_ddl("docs", "euclidean")
    assert idx_name.startswith('"IVF_')
    assert "DISTANCE euclidean" in ddl
    assert "neighbor partitions 32" in ddl

    _, ddl = ds._get_ivf_index_ddl(
        "docs",
        "euclidean",
        {
            "idx_type": "IVF",
            "neighbor_partitions": 64,
            "samples_per_partition": 8,
            "min_vectors_per_partition": 2,
        },
    )
    assert "neighbor partitions 64" in ddl
    assert "samples_per_partition 8" in ddl
    assert "min_vectors_per_partition 2" in ddl

    with pytest.raises(ValueError, match="Invalid parameter"):
        ds._get_ivf_index_ddl("docs", "dot", {"idx_type": "IVF", "bogus": 1})

    with pytest.raises(ValueError, match="Invalid parameter"):
        ds._get_ivf_index_ddl("docs", "dot", {"idx_type": "IVF", "neighbor_part": 32})

    with pytest.raises(ValueError, match="parallel must be an integer"):
        ds._get_ivf_index_ddl("docs", "dot", {"idx_type": "IVF", "parallel": "8 NOLOGGING"})


def test_create_index_helpers(monkeypatch):
    executed = []

    def fake_index_exists(connection, idx_name, table_name=None):
        return False

    monkeypatch.setattr(ds, "_index_exists", fake_index_exists)

    cursor = _ExecCursor()
    ds._create_hnsw_index(_Connection(cursor), "docs", "cosine", {"idx_type": "HNSW", "idx_name": '"IDX"'})
    executed.extend(cursor.executed)
    assert executed

    cursor = _ExecCursor()
    ds._create_ivf_index(_Connection(cursor), "docs", "dot", {"idx_type": "IVF", "idx_name": '"IDX"'})
    assert cursor.executed

    monkeypatch.setattr(ds, "_index_exists", lambda *args, **kwargs: True)
    cursor = _ExecCursor()
    ds._create_hnsw_index(_Connection(cursor), "docs", "cosine", {"idx_type": "HNSW", "idx_name": '"IDX"'})
    assert cursor.executed == []

    cursor = _ExecCursor()
    ds._create_ivf_index(_Connection(cursor), "docs", "cosine", {"idx_type": "IVF", "idx_name": '"IDX"'})
    assert cursor.executed == []


@pytest.mark.asyncio
async def test_create_index_helpers_async(monkeypatch):
    async def fake_index_exists_async(*args, **kwargs):
        return False

    monkeypatch.setattr(ds, "_index_exists_async", fake_index_exists_async)

    cursor = _AsyncExecCursor()
    await ds._create_hnsw_index_async(
        _AsyncConnection(cursor), "docs", "cosine", {"idx_type": "HNSW", "idx_name": '"IDX"'}
    )
    assert cursor.executed

    cursor = _AsyncExecCursor()
    await ds._create_ivf_index_async(_AsyncConnection(cursor), "docs", "dot", {"idx_type": "IVF", "idx_name": '"IDX"'})
    assert cursor.executed

    async def fake_index_exists_async_true(*args, **kwargs):
        return True

    monkeypatch.setattr(ds, "_index_exists_async", fake_index_exists_async_true)
    cursor = _AsyncExecCursor()
    await ds._create_hnsw_index_async(
        _AsyncConnection(cursor), "docs", "cosine", {"idx_type": "HNSW", "idx_name": '"IDX"'}
    )
    assert cursor.executed == []

    cursor = _AsyncExecCursor()
    await ds._create_ivf_index_async(_AsyncConnection(cursor), "docs", "dot", {"idx_type": "IVF", "idx_name": '"IDX"'})
    assert cursor.executed == []


def test_document_store_internal_helpers(monkeypatch):
    assert ds._get_document_columns(support_sparse_embeddings=False) == ds.BASE_DOCUMENT_COLUMNS
    assert ds._get_document_columns(support_sparse_embeddings=True)[-1] == "sparse_embedding"
    assert "sparse_embedding" in ds._get_table_dict(4, support_sparse_embeddings=True)
    assert "sparse_embedding" not in ds._get_insert_query("docs", "", support_sparse_embeddings=False)
    assert "sparse_embedding" in ds._get_insert_query("docs", "", support_sparse_embeddings=True)
    assert "sparse_embedding" not in ds._get_merge_query("docs", support_sparse_embeddings=False)
    assert "sparse_embedding" in ds._get_merge_query("docs", support_sparse_embeddings=True)

    assert ds._normalize_sparse_vector_index_config({"enabled": False}) == {"enabled": False}
    assert ds._normalize_sparse_vector_index_config({"enabled": True}) == {
        "enabled": True,
        "distance_strategy": "cosine",
        "params": None,
    }

    with pytest.raises(ValueError, match="Invalid sparse_vector_index parameter"):
        ds._normalize_sparse_vector_index_config({"bad": True})
    with pytest.raises(ValueError, match="must be a boolean"):
        ds._normalize_sparse_vector_index_config({"enabled": "yes"})
    with pytest.raises(ValueError, match="Invalid distance_function"):
        ds._normalize_sparse_vector_index_config({"enabled": True, "distance_strategy": "bad"})
    with pytest.raises(ValueError, match="must be a dictionary"):
        ds._normalize_sparse_vector_index_config({"enabled": True, "params": "bad"})

    sparse = _SparseValue([1], [0.5])
    docs = ds.OracleDocumentStore._get_result_to_documents(
        [
            ("1", "hello", b"blob", {"a": 1}, "text/plain", None, 0.9, [0.1], sparse),
            ("2", "world", None, None, None, {"x": 1}, None, None, None),
        ],
        ["ID", "CONTENT", "BLOB_DATA", "BLOB_META", "BLOB_MIME_TYPE", "META", "SCORE", "EMBEDDING", "SPARSE_EMBEDDING"],
    )
    assert docs[0].blob == ByteStream(b"blob", meta={"a": 1}, mime_type="text/plain")
    assert docs[0].sparse_embedding == SparseEmbedding(indices=[1], values=[0.5])
    assert docs[0].meta == {}
    assert docs[1].blob is None
    assert docs[1].meta == {"x": 1}

    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )

    dense_cursor = _ExecCursor(
        fetchall_result=[("1", "hello", None, None, None, {"topic": "x"}, None, [0.1], None)],
        description=[
            SimpleNamespace(name=name)
            for name in (
                "ID",
                "CONTENT",
                "BLOB_DATA",
                "BLOB_META",
                "BLOB_MIME_TYPE",
                "META",
                "SCORE",
                "EMBEDDING",
                "SPARSE_EMBEDDING",
            )
        ],
    )

    @contextmanager
    def dense_connection(_client):
        yield _Connection(dense_cursor)

    monkeypatch.setattr(ds, "_get_connection", dense_connection)
    monkeypatch.setattr(
        ds, "_get_filter_string", lambda filters, metadata_column, bind_variables: bind_variables.append("x") or "COND"
    )
    store._initialized = True
    store._client = object()
    docs = store._embedding_retrieval([0.1, 0.2], filters={"field": "meta.topic", "operator": "==", "value": "x"})
    assert docs[0].id == "1"
    assert "WHERE COND" in dense_cursor.executed[0][0]
    assert dense_cursor.executed[0][1][0]["value0"] == "x"

    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    store._initialized = True
    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        store._embedding_retrieval(SparseEmbedding(indices=[0], values=[1.0]))


@pytest.mark.asyncio
async def test_document_store_async_internal_helpers(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )

    cursor = _AsyncExecCursor(
        fetchall_result=[("1", "hello", None, None, None, {"topic": "x"}, None, [0.1], _SparseValue([1], [0.5]))],
        description=[
            SimpleNamespace(name=name)
            for name in (
                "ID",
                "CONTENT",
                "BLOB_DATA",
                "BLOB_META",
                "BLOB_MIME_TYPE",
                "META",
                "SCORE",
                "EMBEDDING",
                "SPARSE_EMBEDDING",
            )
        ],
    )

    @asynccontextmanager
    async def async_connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(ds, "_get_connection_async", async_connection)
    monkeypatch.setattr(
        ds, "_get_filter_string", lambda filters, metadata_column, bind_variables: bind_variables.append("x") or "COND"
    )
    store._initialized_async = True
    store._client_async = object()

    docs = await store._embedding_retrieval_async(
        SparseEmbedding(indices=[1], values=[0.5]),
        filters={"field": "meta.topic", "operator": "==", "value": "x"},
    )
    assert docs[0].sparse_embedding == SparseEmbedding(indices=[1], values=[0.5])
    assert "vector_distance(sparse_embedding" in cursor.executed[0][0]
    assert cursor.executed[0][1][0]["value0"] == "x"

    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    store._initialized_async = True
    store._client_async = object()
    monkeypatch.setattr(ds, "_get_connection_async", async_connection)
    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        await store._embedding_retrieval_async(SparseEmbedding(indices=[0], values=[1.0]))


def test_filter_write_and_delete_sync_helpers(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )
    store._initialized = True
    store._client = object()

    cursor = _ExecCursor(
        fetchall_result=[("1", "hello", None, None, None, {"topic": "x"}, None, [0.1], None)],
        description=[
            SimpleNamespace(name=name)
            for name in (
                "ID",
                "CONTENT",
                "BLOB_DATA",
                "BLOB_META",
                "BLOB_MIME_TYPE",
                "META",
                "SCORE",
                "EMBEDDING",
                "SPARSE_EMBEDDING",
            )
        ],
    )
    connection = _Connection(cursor)

    @contextmanager
    def get_connection(_client):
        yield connection

    monkeypatch.setattr(ds, "_get_connection", get_connection)
    monkeypatch.setattr(
        ds, "_get_filter_string", lambda _filters, _meta, bind_variables: bind_variables.append("x") or "COND"
    )

    docs = store.filter_documents({"field": "meta.topic", "operator": "==", "value": "x"})
    assert docs[0].id == "1"
    assert "WHERE COND" in cursor.executed[0][0]
    assert cursor.executed[0][1][0]["value0"] == "x"

    cursor.executed.clear()
    docs = store.filter_documents()
    assert docs[0].id == "1"
    assert "WHERE" not in cursor.executed[0][0]

    monkeypatch.setattr(ds.oracledb, "SparseVector", lambda dim, indices, values: (dim, indices, list(values)))
    doc = Document(id="1", content="hello", embedding=[0.1, 0.2], meta={"a": 1}, score=0.9)
    doc.sparse_embedding = SparseEmbedding(indices=[1], values=[0.5])

    assert store.write_documents([doc], policy=ds.DuplicatePolicy.OVERWRITE) == 1
    assert "MERGE INTO" in cursor.executemany_calls[0][0]
    assert connection.committed is True
    assert len(cursor.executemany_calls[0][1][0]) == 9

    cursor.executemany_calls.clear()
    assert store.write_documents([], policy=ds.DuplicatePolicy.SKIP) == 0
    assert "ignore_row_on_dupkey_index" in cursor.executemany_calls[0][0]

    cursor.executemany_calls.clear()
    dense_doc = Document(id="3", content="dense", embedding=[0.2, 0.3])
    assert store.write_documents([dense_doc], policy=ds.DuplicatePolicy.FAIL) == 1
    assert "INSERT" in cursor.executemany_calls[0][0]
    assert "ignore_row_on_dupkey_index" not in cursor.executemany_calls[0][0]

    store.delete_documents(["1", "2"])
    assert "DELETE FROM" in cursor.executed[-1][0]
    assert cursor.executed[-1][1][0] == {"id1": "1", "id2": "2"}

    bad_store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    bad_store._initialized = True
    bad_store._client = object()
    with pytest.raises(ValueError, match="'documents' must contain a list of Document"):
        bad_store.write_documents([object()])  # type: ignore[list-item]

    sparse_doc = Document(id="2")
    sparse_doc.sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])
    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        bad_store.write_documents([sparse_doc])

    monkeypatch.setattr(ds, "_get_connection", get_connection)
    assert bad_store.write_documents([Document(id="4", content="dense", embedding=[0.1])]) == 1


@pytest.mark.asyncio
async def test_filter_write_and_delete_async_helpers(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )
    store._initialized_async = True
    store._client_async = object()

    cursor = _AsyncExecCursor(
        fetchall_result=[("1", "hello", None, None, None, {"topic": "x"}, None, [0.1], None)],
        description=[
            SimpleNamespace(name=name)
            for name in (
                "ID",
                "CONTENT",
                "BLOB_DATA",
                "BLOB_META",
                "BLOB_MIME_TYPE",
                "META",
                "SCORE",
                "EMBEDDING",
                "SPARSE_EMBEDDING",
            )
        ],
    )
    connection = _AsyncConnection(cursor)

    async def handle_context(callback):
        return await callback(connection)

    monkeypatch.setattr(store, "_handle_context", handle_context)
    monkeypatch.setattr(
        ds, "_get_filter_string", lambda _filters, _meta, bind_variables: bind_variables.append("x") or "COND"
    )

    docs = await store.filter_documents_async({"field": "meta.topic", "operator": "==", "value": "x"})
    assert docs[0].id == "1"
    assert "WHERE COND" in cursor.executed[0][0]
    assert cursor.executed[0][1][0]["value0"] == "x"

    cursor.executed.clear()
    docs = await store.filter_documents_async()
    assert docs[0].id == "1"
    assert "WHERE" not in cursor.executed[0][0]

    monkeypatch.setattr(ds.oracledb, "SparseVector", lambda dim, indices, values: (dim, indices, list(values)))
    doc = Document(id="1", content="hello", embedding=[0.1, 0.2], meta={"a": 1}, score=0.9)
    doc.sparse_embedding = SparseEmbedding(indices=[1], values=[0.5])

    assert await store.write_documents_async([doc], policy=ds.DuplicatePolicy.OVERWRITE) == 1
    assert "MERGE INTO" in cursor.executemany_calls[0][0]
    assert connection.committed is True
    assert len(cursor.executemany_calls[0][1][0]) == 9

    cursor.executemany_calls.clear()
    assert await store.write_documents_async([], policy=ds.DuplicatePolicy.SKIP) == 0
    assert "ignore_row_on_dupkey_index" in cursor.executemany_calls[0][0]

    cursor.executemany_calls.clear()
    dense_doc = Document(id="3", content="dense", embedding=[0.2, 0.3])
    assert await store.write_documents_async([dense_doc], policy=ds.DuplicatePolicy.FAIL) == 1
    assert "INSERT" in cursor.executemany_calls[0][0]
    assert "ignore_row_on_dupkey_index" not in cursor.executemany_calls[0][0]

    await store.delete_documents_async(["1", "2"])
    assert "DELETE FROM" in cursor.executed[-1][0]
    assert cursor.executed[-1][1][0] == {"id1": "1", "id2": "2"}

    bad_store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    bad_store._initialized_async = True
    bad_store._client_async = object()
    monkeypatch.setattr(bad_store, "_handle_context", handle_context)
    assert await bad_store.write_documents_async([Document(id="4", content="dense", embedding=[0.1])]) == 1


def test_ensure_initialized_creates_connection_and_table(monkeypatch):
    connection_params = oracle_unit_test_connection_params()
    store = ds.OracleDocumentStore(connection_params=connection_params, table_name="docs", embedding_dim=4)
    current_version = ds.oracledb.__version__

    created = {"connect": None, "table": 0}

    def fake_connect(**kwargs):
        created["connect"] = kwargs
        return object()

    @contextmanager
    def fake_connection(_client):
        yield _Connection(_ExecCursor())

    monkeypatch.setattr(ds.oracledb, "connect", fake_connect)
    monkeypatch.setattr(ds, "_get_connection", fake_connection)
    monkeypatch.setattr(ds, "_table_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        ds,
        "_create_table",
        lambda *_args, **_kwargs: created.__setitem__("table", created["table"] + 1),
    )
    monkeypatch.setattr(ds.oracledb, "__version__", current_version)

    store._ensure_initialized()
    assert created["connect"] == connection_params
    assert created["table"] == 1
    assert store._initialized is True


@pytest.mark.asyncio
async def test_ensure_initialized_async_creates_connection_and_table(monkeypatch):
    connection_params = oracle_unit_test_connection_params()
    store = ds.OracleDocumentStore(connection_params=connection_params, table_name="docs", embedding_dim=4)
    current_version = ds.oracledb.__version__

    created = {"connect": None, "table": 0}

    async def fake_connect_async(**kwargs):
        created["connect"] = kwargs
        return object()

    @asynccontextmanager
    async def fake_connection(_client):
        yield _AsyncConnection(_AsyncExecCursor())

    async def table_missing(*_args, **_kwargs):
        return False

    async def create_table_async(*_args, **_kwargs):
        created["table"] += 1

    monkeypatch.setattr(ds.oracledb, "connect_async", fake_connect_async)
    monkeypatch.setattr(ds, "_get_connection_async", fake_connection)
    monkeypatch.setattr(ds, "_table_exists_async", table_missing)
    monkeypatch.setattr(ds, "_create_table_async", create_table_async)
    monkeypatch.setattr(ds.oracledb, "__version__", current_version)

    await store._ensure_initialized_async()
    assert created["connect"] == connection_params
    assert created["table"] == 1
    assert store._initialized_async is True


def test_vectorizer_preference_parameter_helpers():
    text_embedder = OracleTextEmbedder(
        connection_params=oracle_unit_test_connection_params(),
        embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
        proxy=None,
        use_connection_pool=False,
    )
    assert ds._get_vectorizer_preference_parameters(text_embedder) == {"model": "ALL_MINILM_L12_V2"}
    assert ds._get_vectorizer_preference_parameters(text_embedder, {"language": "en"}) == {
        "language": "en",
        "model": "ALL_MINILM_L12_V2",
    }
    assert ds._get_vectorizer_preference_parameters(text_embedder, {"model": "ALL_MINILM_L12_V2"}) == {
        "model": "ALL_MINILM_L12_V2"
    }

    external_embedder = OracleTextEmbedder(
        connection_params=oracle_unit_test_connection_params(),
        embedding_params={"provider": "openai", "model": "text-embedding-3-small"},
        proxy=None,
        use_connection_pool=False,
    )
    assert ds._get_vectorizer_preference_parameters(external_embedder) == {
        "embedder_spec": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    assert ds._get_vectorizer_preference_parameters(
        external_embedder,
        {"embedder_spec": {"provider": "openai", "model": "text-embedding-3-small"}},
    ) == {"embedder_spec": {"provider": "openai", "model": "text-embedding-3-small"}}
    assert (
        ds._validate_vectorizer_parameters(
            {"provider": "database", "model": "ALL_MINILM_L12_V2"},
            {"model": "ALL_MINILM_L12_V2"},
        )
        is True
    )
    assert (
        ds._validate_vectorizer_parameters(
            {"provider": "openai", "model": "text-embedding-3-small"},
            {"embedder_spec": {"provider": "openai", "model": "text-embedding-3-small"}},
        )
        is True
    )

    with pytest.raises(ValueError, match="Mismatch between text_embedder"):
        ds._get_vectorizer_preference_parameters(text_embedder, {"model": "OTHER"})

    with pytest.raises(ValueError, match="embedder_spec must exactly match"):
        ds._get_vectorizer_preference_parameters(external_embedder, {"embedder_spec": {"provider": "openai"}})

    with pytest.raises(ValueError, match="text_embedder must be an instance"):
        ds._validate_text_embedder_instance(object())


def test_hybrid_index_ddl_builder():
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
    )
    preference = ds.OracleVectorizerPreference(store, "PREF_X")

    ddl = ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {})
    assert 'CREATE HYBRID VECTOR INDEX "IDX" ON "docs"(content)' in ddl
    assert "PARAMETERS ('vectorizer PREF_X ')" in ddl

    ddl = ds._get_hybrid_index_ddl(
        store._table_name,
        '"IDX"',
        preference,
        {
            "parameters": {"word_min_len": 2},
            "filter_by": ["id", "meta"],
            "order_by": ["id"],
            "order_by_asc": False,
            "parallel": 4,
        },
    )
    assert "word_min_len 2" in ddl
    assert 'FILTER BY "ID","META"' in ddl
    assert 'ORDER BY "ID" DESC' in ddl
    assert "PARALLEL 4" in ddl

    ddl = ds._get_hybrid_index_ddl(
        store._table_name,
        '"IDX"',
        preference,
        {
            "filter_by": ['"id"', "schema.meta", '"Mixed"."CaseCol"'],
            "order_by": ['"ts"'],
        },
    )
    assert 'FILTER BY "id","SCHEMA"."META","Mixed"."CaseCol"' in ddl
    assert 'ORDER BY "ts" ASC' in ddl

    with pytest.raises(ValueError, match="Vectorization parameters must be given"):
        ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {"parameters": {"vectorizer": "PREF_Y"}})

    with pytest.raises(ValueError, match="parallel must be a positive integer"):
        ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {"parallel": "4"})

    with pytest.raises(ValueError, match="filter_by contains an invalid identifier"):
        ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {"filter_by": ["id DESC"]})

    with pytest.raises(ValueError, match="order_by contains an invalid identifier"):
        ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {"order_by": ['id"']})

    with pytest.raises(ValueError, match="order_by_asc must be a boolean"):
        ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {"order_by_asc": "false"})

    with pytest.raises(ValueError, match="parallel must be a positive integer"):
        ds._get_hybrid_index_ddl(store._table_name, '"IDX"', preference, {"parallel": 0})


def test_vectorizer_preference_create_drop_and_hybrid_index(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
    )
    store._initialized = True
    store._client = object()

    text_embedder = OracleTextEmbedder(
        connection_params=oracle_unit_test_connection_params(),
        embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
        proxy=None,
        use_connection_pool=False,
    )

    cursor = _ExecCursor()

    @contextmanager
    def connection(_client):
        yield _Connection(cursor)

    monkeypatch.setattr(ds, "_get_connection", connection)
    monkeypatch.setattr(ds, "_index_exists", lambda *args, **kwargs: False)

    preference = ds.OracleVectorizerPreference.create(store, text_embedder, preference_name="PREF_TEST")
    assert preference.preference_name == "PREF_TEST"
    assert "CREATE_PREFERENCE" in cursor.executed[0][0]
    assert json.loads(cursor.executed[0][1][0][1]) == {"model": "ALL_MINILM_L12_V2"}

    preference.drop()
    assert "DROP_PREFERENCE" in cursor.executed[1][0]

    cursor.executed.clear()
    store.create_hybrid_vector_index("IDX_TEST", vectorizer_preference=preference, params={"parallel": 2})
    assert 'CREATE HYBRID VECTOR INDEX "IDX_TEST"' in cursor.executed[0][0]

    cursor.executed.clear()
    store.create_hybrid_vector_index("IDX_TEXT", text_embedder=text_embedder)
    assert "CREATE_PREFERENCE" in cursor.executed[0][0]
    assert 'CREATE HYBRID VECTOR INDEX "IDX_TEXT"' in cursor.executed[1][0]
    assert "DROP_PREFERENCE" in cursor.executed[2][0]

    with pytest.raises(ValueError, match="Exactly one of 'vectorizer_preference' or 'text_embedder'"):
        store.create_hybrid_vector_index("IDX_BAD")

    with pytest.raises(ValueError, match="Exactly one of 'vectorizer_preference' or 'text_embedder'"):
        store.create_hybrid_vector_index("IDX_BAD", vectorizer_preference=preference, text_embedder=text_embedder)

    monkeypatch.setattr(ds, "_index_exists", lambda *_args, **_kwargs: True)
    cursor.executed.clear()
    store.create_hybrid_vector_index("IDX_EXISTS", vectorizer_preference=preference)
    assert cursor.executed == []

    cursor.executed.clear()
    drops: list[str] = []

    def failing_ddl(*_args, **_kwargs):
        raise ValueError("ddl failed")

    monkeypatch.setattr(ds, "_get_hybrid_index_ddl", failing_ddl)
    monkeypatch.setattr(
        ds.OracleVectorizerPreference,
        "drop",
        lambda self: drops.append(self.preference_name),
    )
    with pytest.raises(ValueError, match="ddl failed"):
        store.create_hybrid_vector_index("IDX_FAIL", text_embedder=text_embedder)
    assert len(drops) == 1

    with pytest.raises(ValueError, match="document_store must be an instance"):
        ds.OracleVectorizerPreference.create(object(), text_embedder)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_vectorizer_preference_create_drop_and_hybrid_index_async(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
    )
    store._initialized_async = True
    store._client_async = object()

    text_embedder = OracleTextEmbedder(
        connection_params=oracle_unit_test_connection_params(),
        embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
        proxy=None,
        use_connection_pool=False,
    )

    cursor = _AsyncExecCursor()

    @asynccontextmanager
    async def connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(ds, "_get_connection_async", connection)

    async def index_missing(*args, **kwargs):
        return False

    monkeypatch.setattr(ds, "_index_exists_async", index_missing)

    preference = await ds.OracleVectorizerPreference.create_async(store, text_embedder, preference_name="PREF_TEST")
    assert preference.preference_name == "PREF_TEST"
    assert "CREATE_PREFERENCE" in cursor.executed[0][0]
    assert json.loads(cursor.executed[0][1][0][1]) == {"model": "ALL_MINILM_L12_V2"}

    await preference.drop_async()
    assert "DROP_PREFERENCE" in cursor.executed[1][0]

    cursor.executed.clear()
    await store.create_hybrid_vector_index_async("IDX_TEST", vectorizer_preference=preference, params={"parallel": 2})
    assert 'CREATE HYBRID VECTOR INDEX "IDX_TEST"' in cursor.executed[0][0]

    cursor.executed.clear()
    await store.create_hybrid_vector_index_async("IDX_TEXT", text_embedder=text_embedder)
    assert "CREATE_PREFERENCE" in cursor.executed[0][0]
    assert 'CREATE HYBRID VECTOR INDEX "IDX_TEXT"' in cursor.executed[1][0]
    assert "DROP_PREFERENCE" in cursor.executed[2][0]

    async def index_exists(*_args, **_kwargs):
        return True

    monkeypatch.setattr(ds, "_index_exists_async", index_exists)
    cursor.executed.clear()
    await store.create_hybrid_vector_index_async("IDX_EXISTS", vectorizer_preference=preference)
    assert cursor.executed == []

    drops: list[str] = []

    async def fake_drop_async(self):
        drops.append(self.preference_name)

    def failing_ddl(*_args, **_kwargs):
        raise ValueError("ddl failed")

    monkeypatch.setattr(ds, "_get_hybrid_index_ddl", failing_ddl)
    monkeypatch.setattr(ds.OracleVectorizerPreference, "drop_async", fake_drop_async)
    with pytest.raises(ValueError, match="ddl failed"):
        await store.create_hybrid_vector_index_async("IDX_FAIL", text_embedder=text_embedder)
    assert len(drops) == 1

    with pytest.raises(ValueError, match="document_store must be an instance"):
        await ds.OracleVectorizerPreference.create_async(object(), text_embedder)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Exactly one of 'vectorizer_preference' or 'text_embedder'"):
        await store.create_hybrid_vector_index_async("IDX_BAD")


def test_to_hybrid_filter():
    assert _to_hybrid_filter({"field": "meta.topic", "operator": "==", "value": "science"}) == {
        "op": "=",
        "path": "meta.topic",
        "type": "string",
        "args": ["science"],
    }
    assert _to_hybrid_filter({"field": "meta.score", "operator": "in", "value": [1, 2]}) == {
        "op": "IN",
        "path": "meta.score",
        "type": "number",
        "args": [1, 2],
    }
    assert _to_hybrid_filter({"field": "meta.score", "operator": "not in", "value": [1, 2]}) == {
        "op": "NOT",
        "args": [{"op": "IN", "path": "meta.score", "type": "number", "args": [1, 2]}],
    }
    assert _to_hybrid_filter(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.topic", "operator": "==", "value": "science"},
                {"field": "meta.score", "operator": ">", "value": 10},
            ],
        }
    ) == {
        "op": "AND",
        "args": [
            {"op": "=", "path": "meta.topic", "type": "string", "args": ["science"]},
            {"op": ">", "path": "meta.score", "type": "number", "args": [10]},
        ],
    }
    assert _to_hybrid_filter({"field": "meta.score", "operator": "!=", "value": 10}) == {
        "op": "!=",
        "path": "meta.score",
        "type": "number",
        "args": [10],
    }
    assert _to_hybrid_filter({"field": "meta.score", "operator": "<=", "value": 10}) == {
        "op": "<=",
        "path": "meta.score",
        "type": "number",
        "args": [10],
    }
    assert _to_hybrid_filter(
        {"operator": "OR", "conditions": [{"field": "meta.topic", "operator": "==", "value": "science"}]}
    ) == {
        "op": "OR",
        "args": [{"op": "=", "path": "meta.topic", "type": "string", "args": ["science"]}],
    }
    assert _to_hybrid_filter(
        {"operator": "NOT", "conditions": [{"field": "meta.topic", "operator": "==", "value": "science"}]}
    ) == {
        "op": "NOT",
        "args": [{"op": "=", "path": "meta.topic", "type": "string", "args": ["science"]}],
    }

    with pytest.raises(FilterError, match="only metadata filters"):
        _to_hybrid_filter({"field": "content", "operator": "==", "value": "science"})
    with pytest.raises(FilterError, match="Invalid metadata key format"):
        _to_hybrid_filter({"field": "meta.bad-key", "operator": "==", "value": "science"})
    with pytest.raises(FilterError, match="not supported for Oracle hybrid retrieval"):
        _to_hybrid_filter({"field": "meta.tags", "operator": "contains", "value": "science"})
    with pytest.raises(FilterError, match="null comparisons"):
        _to_hybrid_filter({"field": "meta.topic", "operator": "==", "value": None})
    with pytest.raises(FilterError, match="Boolean values are not supported"):
        _to_hybrid_filter({"field": "meta.flag", "operator": "==", "value": True})
    with pytest.raises(FilterError, match="non-empty list"):
        _to_hybrid_filter({"field": "meta.score", "operator": "in", "value": []})
    with pytest.raises(FilterError, match="same type"):
        _to_hybrid_filter({"field": "meta.score", "operator": "in", "value": [1, "2"]})
    with pytest.raises(FilterError, match="cannot be used"):
        _to_hybrid_filter({"field": "meta.score", "operator": "like", "value": 1})  # type: ignore[arg-type]
    with pytest.raises(FilterError, match="Invalid operator"):
        _to_hybrid_filter({"operator": "XOR", "conditions": []})
    with pytest.raises(FilterError, match="Filter structure is not correct"):
        _to_hybrid_filter({"field": "meta.topic", "value": "science"})  # type: ignore[arg-type]


def test_embedding_retrieval_sparse_without_filters(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )
    store._initialized = True
    store._client = object()

    cursor = _ExecCursor(
        fetchall_result=[("1", "hello", None, None, None, {"topic": "x"}, None, None, _SparseValue([1], [0.5]))],
        description=[
            SimpleNamespace(name=name)
            for name in (
                "ID",
                "CONTENT",
                "BLOB_DATA",
                "BLOB_META",
                "BLOB_MIME_TYPE",
                "META",
                "SCORE",
                "EMBEDDING",
                "SPARSE_EMBEDDING",
            )
        ],
    )

    @contextmanager
    def dense_connection(_client):
        yield _Connection(cursor)

    monkeypatch.setattr(ds, "_get_connection", dense_connection)
    monkeypatch.setattr(ds.oracledb, "SparseVector", lambda dim, indices, values: (dim, indices, list(values)))

    docs = store._embedding_retrieval(SparseEmbedding(indices=[1], values=[0.5]))
    assert docs[0].sparse_embedding == SparseEmbedding(indices=[1], values=[0.5])
    assert "vector_distance(sparse_embedding" in cursor.executed[0][0]


@pytest.mark.asyncio
async def test_embedding_retrieval_async_dense_without_filters(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )
    store._initialized_async = True
    store._client_async = object()

    cursor = _AsyncExecCursor(
        fetchall_result=[("1", "hello", None, None, None, {"topic": "x"}, None, [0.1], None)],
        description=[
            SimpleNamespace(name=name)
            for name in (
                "ID",
                "CONTENT",
                "BLOB_DATA",
                "BLOB_META",
                "BLOB_MIME_TYPE",
                "META",
                "SCORE",
                "EMBEDDING",
                "SPARSE_EMBEDDING",
            )
        ],
    )

    @asynccontextmanager
    async def async_connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(ds, "_get_connection_async", async_connection)
    docs = await store._embedding_retrieval_async([0.1, 0.2])
    assert docs[0].embedding == [0.1]
    assert "vector_distance(embedding" in cursor.executed[0][0]


def test_store_index_dispatch(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )

    calls = []
    monkeypatch.setattr(ds, "_create_hnsw_index", lambda *args: calls.append(("hnsw", args)))
    monkeypatch.setattr(ds, "_create_ivf_index", lambda *args: calls.append(("ivf", args)))

    store._create_index(_Connection(_ExecCursor()), None)
    assert calls[-1][0] == "hnsw"

    store._create_index(_Connection(_ExecCursor()), {"idx_type": "HNSW", "idx_name": "dense"})
    assert calls[-1][0] == "hnsw"

    store._create_index(_Connection(_ExecCursor()), {"idx_type": "IVF", "idx_name": "dense"})
    assert calls[-1][0] == "ivf"

    with pytest.raises(ValueError, match="Only supported indexes"):
        store._create_index(_Connection(_ExecCursor()), {"idx_type": "BAD"})

    store._create_sparse_index(_Connection(_ExecCursor()), None, "dot")
    assert calls[-1][0] == "hnsw"

    store._create_sparse_index(_Connection(_ExecCursor()), {"idx_type": "HNSW", "idx_name": "sparse"}, "dot")
    assert calls[-1][0] == "hnsw"

    store._create_sparse_index(_Connection(_ExecCursor()), {"idx_type": "IVF", "idx_name": "sparse"}, "dot")
    assert calls[-1][0] == "ivf"

    with pytest.raises(ValueError, match="Only supported indexes"):
        store._create_sparse_index(_Connection(_ExecCursor()), {"idx_type": "BAD"}, "dot")


@pytest.mark.asyncio
async def test_store_index_dispatch_async(monkeypatch):
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )

    calls = []

    async def fake_hnsw(*args):
        calls.append(("hnsw", args))

    async def fake_ivf(*args):
        calls.append(("ivf", args))

    monkeypatch.setattr(ds, "_create_hnsw_index_async", fake_hnsw)
    monkeypatch.setattr(ds, "_create_ivf_index_async", fake_ivf)

    await store._create_index_async(_AsyncConnection(_AsyncExecCursor()), None)
    assert calls[-1][0] == "hnsw"

    await store._create_index_async(_AsyncConnection(_AsyncExecCursor()), {"idx_type": "HNSW", "idx_name": "dense"})
    assert calls[-1][0] == "hnsw"

    await store._create_index_async(_AsyncConnection(_AsyncExecCursor()), {"idx_type": "IVF", "idx_name": "dense"})
    assert calls[-1][0] == "ivf"

    with pytest.raises(ValueError, match="Only supported indexes"):
        await store._create_index_async(_AsyncConnection(_AsyncExecCursor()), {"idx_type": "BAD"})

    await store._create_sparse_index_async(_AsyncConnection(_AsyncExecCursor()), None, "dot")
    assert calls[-1][0] == "hnsw"

    await store._create_sparse_index_async(
        _AsyncConnection(_AsyncExecCursor()), {"idx_type": "HNSW", "idx_name": "sparse"}, "dot"
    )
    assert calls[-1][0] == "hnsw"

    await store._create_sparse_index_async(
        _AsyncConnection(_AsyncExecCursor()), {"idx_type": "IVF", "idx_name": "sparse"}, "dot"
    )
    assert calls[-1][0] == "ivf"

    with pytest.raises(ValueError, match="Only supported indexes"):
        await store._create_sparse_index_async(_AsyncConnection(_AsyncExecCursor()), {"idx_type": "BAD"}, "dot")


def test_filter_helper_error_branches():
    with pytest.raises(ValueError, match="cannot be used"):
        _convert_oper_to_sql("bad", "meta", "meta.key", ":value0")

    with pytest.raises(FilterError, match="Invalid metadata key format"):
        _get_filter_string({"field": "meta.bad-key", "operator": "==", "value": "x"}, "meta", [])

    with pytest.raises(ValueError, match="Invalid operator"):
        _get_filter_string({"operator": "XOR", "conditions": []}, "meta", [])


def test_create_table_and_delete_helpers(monkeypatch):
    monkeypatch.setattr(ds, "_table_exists", lambda *args, **kwargs: False)
    cursor = _ExecCursor()
    connection = _Connection(cursor)
    ds._create_table(connection, '"docs"', 4, support_sparse_embeddings=False)
    assert "CREATE TABLE" in cursor.executed[0][0]

    monkeypatch.setattr(ds, "_table_exists", lambda *args, **kwargs: True)
    cursor = _ExecCursor()
    ds._create_table(_Connection(cursor), '"docs"', 4, support_sparse_embeddings=False)
    assert cursor.executed == []

    ddl, bind_vars = ds._get_delete_ddl('"docs"', ["a", "b"])
    assert ddl == 'DELETE FROM "docs" WHERE id IN (:id1, :id2)'
    assert bind_vars == {"id1": "a", "id2": "b"}

    with pytest.raises(ValueError, match="No ids provided"):
        ds._get_delete_ddl('"docs"', None)


def test_write_documents_sync_empty_input(monkeypatch):
    cursor = _ExecCursor()
    connection = _Connection(cursor)
    store = ds.OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )

    @contextmanager
    def fake_connection(_client):
        yield connection

    monkeypatch.setattr(store, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(ds, "_get_connection", fake_connection)

    assert store.write_documents([]) == 0
    assert cursor.executemany_calls[0][1] == []


@pytest.mark.asyncio
async def test_create_table_async_helper(monkeypatch):
    async def table_missing(*args, **kwargs):
        return False

    monkeypatch.setattr(ds, "_table_exists_async", table_missing)
    cursor = _AsyncExecCursor()
    await ds._create_table_async(_AsyncConnection(cursor), '"docs"', 4, support_sparse_embeddings=False)
    assert "CREATE TABLE" in cursor.executed[0][0]

    async def table_exists(*args, **kwargs):
        return True

    monkeypatch.setattr(ds, "_table_exists_async", table_exists)
    cursor = _AsyncExecCursor()
    await ds._create_table_async(_AsyncConnection(cursor), '"docs"', 4, support_sparse_embeddings=False)
    assert cursor.executed == []


def test_document_store_init_validation_and_sync_initialization(monkeypatch):
    connection_params = oracle_unit_test_connection_params()
    current_version = ds.oracledb.__version__

    with pytest.raises(ValueError, match="Invalid distance_function"):
        ds.OracleDocumentStore(connection_params=connection_params, vector_index_distance_strategy="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid field"):
        ds.OracleDocumentStore(connection_params=connection_params, vector_index_embedding_field="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires support_sparse_embeddings=True"):
        ds.OracleDocumentStore(
            connection_params=connection_params,
            support_sparse_embeddings=False,
            vector_index_embedding_field="sparse_embedding",
        )

    with pytest.raises(ValueError, match="Configure sparse index either"):
        ds.OracleDocumentStore(
            connection_params=connection_params,
            create_vector_index=True,
            vector_index_embedding_field="sparse_embedding",
            sparse_vector_index={"enabled": True},
        )

    store = ds.OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        create_vector_index=True,
        sparse_vector_index={"enabled": True, "distance_strategy": "dot"},
    )

    created = {"index": 0, "sparse": 0}
    monkeypatch.setattr(ds.oracledb, "create_pool", lambda **kwargs: object())
    monkeypatch.setattr(ds, "_table_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        store, "_create_index", lambda *args, **kwargs: created.__setitem__("index", created["index"] + 1)
    )
    monkeypatch.setattr(
        store,
        "_create_sparse_index",
        lambda *args, **kwargs: created.__setitem__("sparse", created["sparse"] + 1),
    )

    @contextmanager
    def fake_connection(_client):
        yield _Connection(_ExecCursor())

    monkeypatch.setattr(ds, "_get_connection", fake_connection)
    store._use_connection_pool = True
    store._ensure_initialized()
    assert created == {"index": 1, "sparse": 1}
    assert store._initialized is True

    old_store = ds.OracleDocumentStore(connection_params=connection_params)
    monkeypatch.setattr(ds.oracledb, "__version__", "2.1.0")
    with pytest.raises(Exception, match="must be >=2.2.0"):
        old_store._ensure_initialized()
    monkeypatch.setattr(ds.oracledb, "__version__", current_version)


@pytest.mark.asyncio
async def test_document_store_async_init_and_write_branches(monkeypatch):
    connection_params = oracle_unit_test_connection_params()
    current_version = ds.oracledb.__version__
    store = ds.OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        create_vector_index=True,
        sparse_vector_index={"enabled": True, "distance_strategy": "dot"},
        use_connection_pool=True,
        support_sparse_embeddings=True,
    )

    async def fake_create_pool_async(**kwargs):
        return object()

    created = {"index": 0, "sparse": 0}
    monkeypatch.setattr(ds.oracledb, "create_pool_async", fake_create_pool_async)

    async def fake_table_exists_async(*args, **kwargs):
        return True

    monkeypatch.setattr(ds, "_table_exists_async", fake_table_exists_async)

    async def fake_create_index_async(*args, **kwargs):
        created["index"] += 1

    async def fake_create_sparse_index_async(*args, **kwargs):
        created["sparse"] += 1

    monkeypatch.setattr(store, "_create_index_async", fake_create_index_async)
    monkeypatch.setattr(store, "_create_sparse_index_async", fake_create_sparse_index_async)

    @asynccontextmanager
    async def fake_connection_async(_client):
        yield _AsyncConnection(_AsyncExecCursor())

    monkeypatch.setattr(ds, "_get_connection_async", fake_connection_async)
    await store._ensure_initialized_async()
    assert created == {"index": 1, "sparse": 1}
    assert store._initialized_async is True

    old_store = ds.OracleDocumentStore(connection_params=connection_params)
    monkeypatch.setattr(ds.oracledb, "__version__", "2.1.0")
    with pytest.raises(Exception, match="must be >=2.2.0"):
        await old_store._ensure_initialized_async()
    monkeypatch.setattr(ds.oracledb, "__version__", current_version)

    cursor = _AsyncExecCursor()
    connection = _AsyncConnection(cursor)

    async def handle_context(callback):
        return await callback(connection)

    store = ds.OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )

    async def ensure_initialized_async():
        return None

    monkeypatch.setattr(store, "_ensure_initialized_async", ensure_initialized_async)
    monkeypatch.setattr(store, "_handle_context", handle_context)
    monkeypatch.setattr(ds.oracledb, "SparseVector", lambda dim, indices, values: (dim, indices, list(values)))

    sparse_doc = Document(id="1", content="hello", embedding=[0.1, 0.2], meta={"a": 1}, score=0.9)
    sparse_doc.sparse_embedding = SparseEmbedding(indices=[1], values=[0.5])
    assert await store.write_documents_async([sparse_doc], policy=ds.DuplicatePolicy.OVERWRITE) == 1
    assert "MERGE INTO" in cursor.executemany_calls[0][0]
    assert connection.committed is True
    assert len(cursor.executemany_calls[0][1][0]) == 9

    cursor = _AsyncExecCursor()
    connection = _AsyncConnection(cursor)
    assert await store.write_documents_async([], policy=ds.DuplicatePolicy.SKIP) == 0
    assert "ignore_row_on_dupkey_index" in cursor.executemany_calls[0][0]

    bad_store = ds.OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    monkeypatch.setattr(bad_store, "_ensure_initialized_async", ensure_initialized_async)
    monkeypatch.setattr(bad_store, "_handle_context", handle_context)
    with pytest.raises(ValueError, match="'documents' must contain a list of Document"):
        await bad_store.write_documents_async([object()])  # type: ignore[list-item]

    sparse_doc = Document(id="2")
    sparse_doc.sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])
    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        await bad_store.write_documents_async([sparse_doc])

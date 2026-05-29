# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import oracledb
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.errors import FilterError

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.embedders.oracle import OracleTextEmbedder
from haystack_integrations.components.retrievers.oracle import OracleHybridRetriever

from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connect_dsn,
    oracle_test_connection_params,
    oracle_unit_test_connection_params,
)


class _HybridCursor:
    def __init__(self, search_rows, rowid_rows, description):
        self.search_rows = search_rows
        self.rowid_rows = rowid_rows
        self.description = description
        self.executed = []
        self.outputtypehandler = None
        self._fetchone_result = None
        self._fetchall_result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def setinputsizes(self, *args, **kwargs):
        return None

    def execute(self, query, *args, **kwargs):
        self.executed.append((query, args, kwargs))
        if "DBMS_HYBRID_VECTOR.SEARCH" in query:
            self._fetchone_result = (json.dumps(self.search_rows),)
        elif "WHERE rowid = :rid" in query:
            self._fetchall_result = [self.rowid_rows[kwargs["rid"]]]

    def fetchone(self):
        return self._fetchone_result

    def fetchall(self):
        return self._fetchall_result


class _AsyncHybridCursor(_HybridCursor):
    async def execute(self, query, *args, **kwargs):
        super().execute(query, *args, **kwargs)

    async def fetchone(self):
        return self._fetchone_result

    async def fetchall(self):
        return self._fetchall_result


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _AsyncConnection(_Connection):
    pass


class _SyncReadable:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


class _AsyncReadable:
    def __init__(self, value):
        self.value = value

    async def read(self):
        return self.value


def _make_store() -> OracleDocumentStore:
    return OracleDocumentStore(
        connection_params=oracle_unit_test_connection_params(),
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=True,
    )


def _drop_table_purge(connection: oracledb.Connection, table_name: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" PURGE')


def test_hybrid_retriever_init_validation_and_params():
    store = _make_store()
    retriever = OracleHybridRetriever(document_store=store, idx_name="IDX_HYB")
    assert retriever.idx_name == '"IDX_HYB"'
    assert retriever.search_mode == "hybrid"
    assert retriever.top_k == 10
    assert retriever.params == {}

    with pytest.raises(ValueError, match="document_store must be an instance"):
        OracleHybridRetriever(document_store=object(), idx_name="IDX_HYB")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="search_mode must be one of"):
        OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", search_mode="bad")  # type: ignore[arg-type]

    for bad_top_k in ("1 ROWS ONLY --", True, False, 0, -1):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", top_k=bad_top_k)  # type: ignore[arg-type]

    for params in (
        {"search_text": "x"},
        {"return": {"topN": 3}},
        {"vector": {"search_text": "x"}},
        {"vector": {"search_vector": [0.1]}},
        {"text": {"search_text": "x"}},
        {"text": {"contains": "x"}},
        {"text": {"json_textcontains": {"x": 1}}},
    ):
        with pytest.raises(ValueError):
            OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", params=params)


def test_hybrid_retriever_search_params():
    retriever = OracleHybridRetriever(
        document_store=_make_store(),
        idx_name="IDX_HYB",
        params={"search_fusion": "UNION", "vector": {"result_max": 20}},
    )
    params = retriever._get_search_params(
        "hello",
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=3,
    )
    assert params["hybrid_index_name"] == '"IDX_HYB"'
    assert params["vector"]["search_text"] == "hello"
    assert params["text"]["search_text"] == "hello"
    assert params["vector"]["result_max"] == 20
    assert params["filter_by"] == {"op": "=", "path": "meta.topic", "type": "string", "args": ["science"]}
    assert params["return"] == {"topN": 3, "values": ["rowid", "score", "vector_score", "text_score"], "format": "JSON"}

    for bad_top_k in ("1 ROWS ONLY --", True, False, 0, -1):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retriever._get_search_params("hello", top_k=bad_top_k)  # type: ignore[arg-type]

    semantic = OracleHybridRetriever(document_store=_make_store(), idx_name="IDX_HYB", search_mode="semantic")
    semantic_params = semantic._get_search_params("hello")
    assert "vector" in semantic_params and "text" not in semantic_params

    keyword = OracleHybridRetriever(document_store=_make_store(), idx_name="IDX_HYB", search_mode="keyword")
    keyword_params = keyword._get_search_params("hello")
    assert "text" in keyword_params and "vector" not in keyword_params

    with pytest.raises(FilterError, match="Cannot combine Haystack filters with params\\['filter_by'\\]"):
        OracleHybridRetriever(document_store=_make_store(), idx_name="IDX_HYB", params={"filter_by": {"op": "="}})._get_search_params(  # noqa: E501
            "hello",
            filters={"field": "meta.topic", "operator": "==", "value": "science"},
        )


def test_hybrid_retriever_decode_helpers_and_no_score_metadata():
    retriever = OracleHybridRetriever(document_store=_make_store(), idx_name="IDX_HYB")
    assert retriever._decode_search_result(_SyncReadable(json.dumps([{"rowid": "1"}]))) == [{"rowid": "1"}]


@pytest.mark.asyncio
async def test_hybrid_retriever_decode_async_helper():
    retriever = OracleHybridRetriever(document_store=_make_store(), idx_name="IDX_HYB")
    assert await retriever._decode_search_result_async(_AsyncReadable(json.dumps([{"rowid": "1"}]))) == [{"rowid": "1"}]
    assert await retriever._decode_search_result_async(_SyncReadable(json.dumps([{"rowid": "2"}]))) == [{"rowid": "2"}]


def test_hybrid_retriever_run(monkeypatch):
    store = _make_store()
    store._initialized = True
    store._client = object()

    search_rows = [{"rowid": "AAABBB", "score": 0.91, "text_score": 0.8, "vector_score": 0.7}]
    rowid_rows = {
        "AAABBB": ("1", "hello", None, None, None, {"topic": "science"}, None, [0.1, 0.2], None),
    }
    description = [
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
    ]
    cursor = _HybridCursor(search_rows, rowid_rows, description)

    @contextmanager
    def connection(_client):
        yield _Connection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.retrievers.oracle.hybrid_retriever._get_connection",
        connection,
    )

    retriever = OracleHybridRetriever(
        document_store=store,
        idx_name="IDX_HYB",
        return_scores=True,
        filters={"field": "meta.kind", "operator": "==", "value": "base"},
        filter_policy=FilterPolicy.MERGE,
    )
    result = retriever.run(
        query="hello",
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=2,
        params={"vector": {"result_max": 8}},
    )

    docs = result["documents"]
    assert docs == [
        Document(
            id="1",
            content="hello",
            meta={"topic": "science", "score": 0.91, "text_score": 0.8, "vector_score": 0.7},
            embedding=[0.1, 0.2],
            score=0.91,
        )
    ]
    assert "DBMS_HYBRID_VECTOR.SEARCH" in cursor.executed[0][0]
    search_params = cursor.executed[0][2]["search_params"]
    assert search_params["return"]["topN"] == 2
    assert search_params["filter_by"]["op"] == "AND"
    assert search_params["vector"]["result_max"] == 8


def test_hybrid_retriever_run_without_scores_and_without_rows(monkeypatch):
    store = _make_store()
    store._initialized = True
    store._client = object()

    empty_cursor = _HybridCursor([], {}, [])

    @contextmanager
    def connection(_client):
        yield _Connection(empty_cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.retrievers.oracle.hybrid_retriever._get_connection",
        connection,
    )

    retriever = OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", return_scores=False)
    result = retriever.run(query="hello")
    assert result == {"documents": []}


def test_hybrid_retriever_run_without_scores_with_rows(monkeypatch):
    store = _make_store()
    store._initialized = True
    store._client = object()

    search_rows = [
        {"rowid": "AAABBB", "score": 0.91, "text_score": 0.8, "vector_score": 0.7},
        {"rowid": "CCCDDD", "score": 0.81, "text_score": 0.7, "vector_score": 0.6},
    ]
    rowid_rows = {
        "AAABBB": ("1", "hello", None, None, None, {"topic": "science"}, None, [0.1, 0.2], None),
        "CCCDDD": ("2", "world", None, None, None, {"topic": "db"}, None, [0.2, 0.3], None),
    }
    description = [
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
    ]
    cursor = _HybridCursor(search_rows, rowid_rows, description)

    @contextmanager
    def connection(_client):
        yield _Connection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.retrievers.oracle.hybrid_retriever._get_connection",
        connection,
    )

    retriever = OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", return_scores=False)
    result = retriever.run(query="hello")
    assert [doc.id for doc in result["documents"]] == ["1", "2"]
    assert "score" not in result["documents"][0].meta


@pytest.mark.asyncio
async def test_hybrid_retriever_run_async(monkeypatch):
    store = _make_store()
    store._initialized_async = True
    store._client_async = object()

    search_rows = [{"rowid": "AAABBB", "score": 0.91, "text_score": 0.8, "vector_score": 0.7}]
    rowid_rows = {
        "AAABBB": ("1", "hello", None, None, None, {"topic": "science"}, None, [0.1, 0.2], None),
    }
    description = [
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
    ]
    cursor = _AsyncHybridCursor(search_rows, rowid_rows, description)

    @asynccontextmanager
    async def connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.retrievers.oracle.hybrid_retriever._get_connection_async",
        connection,
    )

    retriever = OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", search_mode="semantic")
    result = await retriever.run_async(query="hello")

    docs = result["documents"]
    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].score == 0.91
    search_params = cursor.executed[0][2]["search_params"]
    assert "text" not in search_params
    assert search_params["vector"]["search_text"] == "hello"


@pytest.mark.asyncio
async def test_hybrid_retriever_run_async_with_scores(monkeypatch):
    store = _make_store()
    store._initialized_async = True
    store._client_async = object()

    search_rows = [{"rowid": "AAABBB", "score": 0.91, "text_score": 0.8, "vector_score": 0.7}]
    rowid_rows = {
        "AAABBB": ("1", "hello", None, None, None, {"topic": "science"}, None, [0.1, 0.2], None),
    }
    description = [
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
    ]
    cursor = _AsyncHybridCursor(search_rows, rowid_rows, description)

    @asynccontextmanager
    async def connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.retrievers.oracle.hybrid_retriever._get_connection_async",
        connection,
    )

    retriever = OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", return_scores=True)
    result = await retriever.run_async(query="hello")
    assert result["documents"][0].meta["score"] == 0.91
    assert result["documents"][0].meta["text_score"] == 0.8
    assert result["documents"][0].meta["vector_score"] == 0.7


@pytest.mark.asyncio
async def test_hybrid_retriever_run_async_without_scores_with_rows(monkeypatch):
    store = _make_store()
    store._initialized_async = True
    store._client_async = object()

    search_rows = [
        {"rowid": "AAABBB", "score": 0.91, "text_score": 0.8, "vector_score": 0.7},
        {"rowid": "CCCDDD", "score": 0.81, "text_score": 0.7, "vector_score": 0.6},
    ]
    rowid_rows = {
        "AAABBB": ("1", "hello", None, None, None, {"topic": "science"}, None, [0.1, 0.2], None),
        "CCCDDD": ("2", "world", None, None, None, {"topic": "db"}, None, [0.2, 0.3], None),
    }
    description = [
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
    ]
    cursor = _AsyncHybridCursor(search_rows, rowid_rows, description)

    @asynccontextmanager
    async def connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.retrievers.oracle.hybrid_retriever._get_connection_async",
        connection,
    )

    retriever = OracleHybridRetriever(document_store=store, idx_name="IDX_HYB", return_scores=False)
    result = await retriever.run_async(query="hello")
    assert [doc.id for doc in result["documents"]] == ["1", "2"]
    assert "score" not in result["documents"][0].meta


def test_hybrid_retriever_to_dict_and_from_dict():
    retriever = OracleHybridRetriever(
        document_store=_make_store(),
        idx_name="IDX_HYB",
        search_mode="keyword",
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=3,
        params={"search_fusion": "UNION"},
        return_scores=True,
        filter_policy=FilterPolicy.MERGE,
    )

    restored = OracleHybridRetriever.from_dict(retriever.to_dict())

    assert isinstance(restored.document_store, OracleDocumentStore)
    assert restored.idx_name == '"IDX_HYB"'
    assert restored.search_mode == "keyword"
    assert restored.filters == {"field": "meta.topic", "operator": "==", "value": "science"}
    assert restored.top_k == 3
    assert restored.params == {"search_fusion": "UNION"}
    assert restored.return_scores is True
    assert restored.filter_policy == FilterPolicy.MERGE

    data = retriever.to_dict()
    del data["init_parameters"]["filter_policy"]
    restored_default = OracleHybridRetriever.from_dict(data)
    assert restored_default.filter_policy == FilterPolicy.REPLACE


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.integration
def test_hybrid_retriever_integration():
    table_name = "mytable_haystack_hybrid"
    connection_params = oracle_test_connection_params()
    connection = oracledb.connect(**connection_params)
    _drop_table_purge(connection, table_name)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name=table_name,
        support_sparse_embeddings=False,
    )
    text_embedder = OracleTextEmbedder(
        connection_params={"dsn": oracle_test_connect_dsn()},
        embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
        proxy=None,
        use_connection_pool=False,
    )

    try:
        store.write_documents(
            [
                Document(id="1", content="Oracle database vector search and semantic retrieval.", meta={"topic": "db"}),
                Document(id="2", content="Bananas are yellow tropical fruits.", meta={"topic": "fruit"}),
                Document(id="3", content="Python supports generators and async functions.", meta={"topic": "code"}),
            ]
        )
        store.create_hybrid_vector_index("hybrid_idx_docs", text_embedder=text_embedder)

        retriever = OracleHybridRetriever(
            document_store=store,
            idx_name="hybrid_idx_docs",
            search_mode="hybrid",
            top_k=2,
            return_scores=True,
        )
        result = retriever.run(
            query="oracle vector database",
            filters={"field": "meta.topic", "operator": "==", "value": "db"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].id == "1"
        assert result["documents"][0].score is not None
        assert "vector_score" in result["documents"][0].meta
        assert "text_score" in result["documents"][0].meta
    finally:
        _drop_table_purge(connection, table_name)
        connection.close()

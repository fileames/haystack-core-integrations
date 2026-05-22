# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import oracledb
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.retrievers.oracle import OracleTextRetriever

from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connection_params,
    oracle_unit_test_connection_params,
)


class _TextCursor:
    def __init__(self, rows, description):
        self.rows = rows
        self.description = description
        self.executed = []
        self.outputtypehandler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, *args, **kwargs):
        self.executed.append((query, args, kwargs))

    def fetchall(self):
        return self.rows


class _AsyncTextCursor(_TextCursor):
    async def execute(self, query, *args, **kwargs):
        self.executed.append((query, args, kwargs))

    async def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _AsyncConnection(_Connection):
    pass


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


@pytest.fixture  # type: ignore
def text_document_store():
    connection_params = oracle_test_connection_params()
    connection = oracledb.connect(**connection_params)
    table_name = "mytable_haystack_text_retriever"

    _drop_table_purge(connection, table_name)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name=table_name,
        embedding_dim=4,
        support_sparse_embeddings=False,
        create_vector_index=False,
    )

    yield store

    _drop_table_purge(connection, table_name)
    connection.close()


def test_generate_accum_query_and_prepare_query():
    assert OracleTextRetriever._prepare_query("refund policy") == '"refund" ACCUM "policy"'
    assert OracleTextRetriever._prepare_query("refund policy", fuzzy=True) == 'fuzzy("refund") ACCUM fuzzy("policy")'
    assert OracleTextRetriever._prepare_query("refund NEAR policy", operator_search=True) == "refund NEAR policy"

    with pytest.raises(ValueError, match="must not be empty"):
        OracleTextRetriever._prepare_query("  ")

    with pytest.raises(ValueError, match="searchable token"):
        OracleTextRetriever._prepare_query("!!!")

    with pytest.raises(TypeError, match="expects query to be a string"):
        OracleTextRetriever._prepare_query(1)  # type: ignore[arg-type]


def test_init_and_serialization_roundtrip():
    store = _make_store()
    retriever = OracleTextRetriever(
        document_store=store,
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=7,
        fuzzy=True,
        operator_search=False,
        return_scores=True,
        filter_policy=FilterPolicy.MERGE,
    )

    assert retriever.top_k == 7
    assert retriever.fuzzy is True
    assert retriever.return_scores is True

    restored = OracleTextRetriever.from_dict(retriever.to_dict())
    assert isinstance(restored.document_store, OracleDocumentStore)
    assert restored.filters == {"field": "meta.topic", "operator": "==", "value": "science"}
    assert restored.top_k == 7
    assert restored.fuzzy is True
    assert restored.operator_search is False
    assert restored.return_scores is True
    assert restored.filter_policy == FilterPolicy.MERGE

    with pytest.raises(ValueError, match="document_store must be an instance"):
        OracleTextRetriever(document_store=object())  # type: ignore[arg-type]


def test_run_delegates_to_document_store():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._text_retrieval.return_value = []

    retriever = OracleTextRetriever(document_store=mock_store, fuzzy=True)
    filters = {"field": "meta.topic", "operator": "==", "value": "foo"}

    result = retriever.run(query="refund policy", filters=filters, top_k=5)

    mock_store._text_retrieval.assert_called_once_with(
        query='fuzzy("refund") ACCUM fuzzy("policy")',
        filters=filters,
        top_k=5,
    )
    assert result == {"documents": []}


@pytest.mark.asyncio
async def test_run_async_delegates_to_document_store():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._text_retrieval_async = AsyncMock(return_value=[])

    retriever = OracleTextRetriever(document_store=mock_store, operator_search=True)
    filters = {"field": "meta.topic", "operator": "==", "value": "foo"}

    result = await retriever.run_async(query="refund NEAR policy", filters=filters, top_k=5)

    mock_store._text_retrieval_async.assert_awaited_once_with(
        query="refund NEAR policy",
        filters=filters,
        top_k=5,
    )
    assert result == {"documents": []}


def test_run_merges_filters_and_return_scores():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._text_retrieval.return_value = [Document(content="hello", meta={}, score=0.8)]

    retriever = OracleTextRetriever(
        document_store=mock_store,
        filters={"field": "meta.kind", "operator": "==", "value": "base"},
        return_scores=True,
        filter_policy=FilterPolicy.MERGE,
    )
    result = retriever.run(
        query="hello",
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
    )

    passed_filters = mock_store._text_retrieval.call_args.kwargs["filters"]
    assert passed_filters["operator"] == "AND"
    assert result["documents"][0].meta["score"] == 0.8


def test_text_retrieval_sync(monkeypatch):
    store = _make_store()
    store._initialized = True
    store._client = object()

    rows = [
        ("1", "hello world", None, None, None, {"topic": "science"}, [0.1], None, 12.5),
    ]
    description = [
        SimpleNamespace(name=name)
        for name in (
            "ID",
            "CONTENT",
            "BLOB_DATA",
            "BLOB_META",
            "BLOB_MIME_TYPE",
            "META",
            "EMBEDDING",
            "SPARSE_EMBEDDING",
            "SCORE",
        )
    ]
    cursor = _TextCursor(rows, description)

    @contextmanager
    def connection(_client):
        yield _Connection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.document_stores.oracle.document_store._get_connection",
        connection,
    )

    docs = store._text_retrieval(
        query='"hello"',
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=2,
    )

    assert docs == [Document(id="1", content="hello world", meta={"topic": "science"}, embedding=[0.1], score=12.5)]
    assert "CONTAINS(content, :query, 1) > 0" in cursor.executed[0][0]
    assert cursor.executed[0][1][0]["query"] == '"hello"'


@pytest.mark.asyncio
async def test_text_retrieval_async(monkeypatch):
    store = _make_store()
    store._initialized_async = True
    store._client_async = object()

    rows = [
        ("1", "hello world", None, None, None, {"topic": "science"}, [0.1], None, 12.5),
    ]
    description = [
        SimpleNamespace(name=name)
        for name in (
            "ID",
            "CONTENT",
            "BLOB_DATA",
            "BLOB_META",
            "BLOB_MIME_TYPE",
            "META",
            "EMBEDDING",
            "SPARSE_EMBEDDING",
            "SCORE",
        )
    ]
    cursor = _AsyncTextCursor(rows, description)

    @asynccontextmanager
    async def connection(_client):
        yield _AsyncConnection(cursor)

    monkeypatch.setattr(
        "haystack_integrations.components.document_stores.oracle.document_store._get_connection_async",
        connection,
    )

    docs = await store._text_retrieval_async(
        query='"hello"',
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=2,
    )

    assert docs == [Document(id="1", content="hello world", meta={"topic": "science"}, embedding=[0.1], score=12.5)]
    assert "CONTAINS(content, :query, 1) > 0" in cursor.executed[0][0]
    assert cursor.executed[0][1][0]["query"] == '"hello"'


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.integration
def test_run_text_retrieval_integration(text_document_store: OracleDocumentStore):
    docs = [
        Document(id="1", content="refund policy for premium plan"),
        Document(id="2", content="shipping policy for standard orders"),
        Document(id="3", content="completely unrelated content"),
    ]
    text_document_store.write_documents(docs)
    text_document_store.create_text_index("oracle_text_idx")

    retriever = OracleTextRetriever(document_store=text_document_store, top_k=2, return_scores=True)
    result = retriever.run(query="refund policy")

    assert result["documents"]
    assert result["documents"][0].id == "1"
    assert result["documents"][0].score is not None
    assert result["documents"][0].meta["score"] == result["documents"][0].score

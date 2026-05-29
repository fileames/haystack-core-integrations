# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import oracledb
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever

from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connection_params,
    oracle_unit_test_connection_params,
)


def drop_table_purge(connection: oracledb.Connection, table_name: str) -> None:
    ddl = f'DROP TABLE IF EXISTS "{table_name}" PURGE'
    with connection.cursor() as cursor:
        cursor.execute(ddl)


@pytest.fixture  # type: ignore
def document_store():
    """
    Create a fresh OracleDocumentStore for tests and purge the table before/after.
    Use a small embedding dimension for concise tests.
    """
    connection_params = oracle_test_connection_params()
    connection = oracledb.connect(**connection_params)
    table_name = "mytable_haystack_retriever"

    drop_table_purge(connection, table_name)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name=table_name,
        embedding_dim=4,
        support_sparse_embeddings=False,
        create_vector_index=False,
    )

    yield store

    drop_table_purge(connection, table_name)
    connection.close()


def test_init_validation():
    """
    Basic init and validation without hitting the database (unit test).
    """
    mock_store = Mock(spec=OracleDocumentStore)

    # valid default
    retriever = OracleEmbeddingRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.distance_strategy == "cosine"

    # invalid distance strategy
    with pytest.raises(ValueError):
        OracleEmbeddingRetriever(document_store=mock_store, distance_strategy="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="document_store must be an instance"):
        OracleEmbeddingRetriever(document_store=object())  # type: ignore[arg-type]


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.parametrize("distance_strategy", ["cosine", "dot", "euclidean"])
@pytest.mark.integration
def test_run_embedding_retrieval(document_store: OracleDocumentStore, distance_strategy: str):
    """
    Integration test for synchronous retrieval.
    Writes a small set of documents with deterministic embeddings and checks nearest neighbors.
    """
    docs = [
        Document(id="1", content="first", embedding=[1.0, 0.0, 0.0, 0.0]),
        Document(id="2", content="second", embedding=[0.9, 0.0, 0.0, 0.0]),
        Document(id="3", content="third", embedding=[0.0, 1.0, 0.0, 0.0]),
    ]
    num_written = document_store.write_documents(docs)
    assert num_written == 3

    retriever = OracleEmbeddingRetriever(document_store=document_store, top_k=2)

    query_embedding = [1.0, 0.0, 0.0, 0.0]
    res = retriever.run(query_embedding=query_embedding, distance_strategy=distance_strategy)
    out_docs = res["documents"]
    assert len(out_docs) == 2
    assert sorted([d.id for d in out_docs]) == ["1", "2"]


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.parametrize("distance_strategy", ["cosine", "dot", "euclidean"])
@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_async_embedding_retrieval(document_store: OracleDocumentStore, distance_strategy: str):
    """
    Integration test for asynchronous retrieval.
    """
    docs = [
        Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0, 0.0]),
        Document(id="b", content="beta", embedding=[0.9, 0.0, 0.0, 0.0]),
        Document(id="c", content="gamma", embedding=[0.0, 1.0, 0.0, 0.0]),
    ]
    num_written = document_store.write_documents(docs)
    assert num_written == 3

    retriever = OracleEmbeddingRetriever(document_store=document_store, top_k=2)

    query_embedding = [1.0, 0.0, 0.0, 0.0]
    res = await retriever.run_async(query_embedding=query_embedding, distance_strategy=distance_strategy)
    out_docs = res["documents"]
    assert len(out_docs) == 2
    assert sorted([d.id for d in out_docs]) == ["a", "b"]


def test_run_delegates_to_document_store():
    """
    Unit test: ensure run() delegates to the document store with expected arguments.
    """
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval.return_value = []

    retriever = OracleEmbeddingRetriever(document_store=mock_store)
    filters = {"field": "meta.topic", "operator": "==", "value": "foo"}
    res = retriever.run(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters=filters,
        top_k=5,
        distance_strategy="cosine",
    )

    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters=filters,
        top_k=5,
        distance_strategy="cosine",
    )
    assert "documents" in res
    assert isinstance(res["documents"], list)


@pytest.mark.parametrize("bad_top_k", ["1 ROWS ONLY --", True, False, 0, -1])
def test_run_rejects_invalid_top_k(bad_top_k):
    mock_store = Mock(spec=OracleDocumentStore)
    retriever = OracleEmbeddingRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.run(query_embedding=[0.1, 0.2], top_k=bad_top_k)

    mock_store._embedding_retrieval.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_top_k", ["1 ROWS ONLY --", True, False, 0, -1])
async def test_run_async_rejects_invalid_top_k(bad_top_k):
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval_async = AsyncMock(return_value=[])
    retriever = OracleEmbeddingRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        await retriever.run_async(query_embedding=[0.1, 0.2], top_k=bad_top_k)

    mock_store._embedding_retrieval_async.assert_not_awaited()


@pytest.mark.parametrize("bad_top_k", ["1 ROWS ONLY --", True, False, 0, -1])
def test_init_rejects_invalid_top_k(bad_top_k):
    mock_store = Mock(spec=OracleDocumentStore)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        OracleEmbeddingRetriever(document_store=mock_store, top_k=bad_top_k)


def test_run_uses_instance_distance_strategy_when_not_overridden():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval.return_value = []

    retriever = OracleEmbeddingRetriever(document_store=mock_store, distance_strategy="dot")
    retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])

    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters={},
        top_k=10,
        distance_strategy="dot",
    )


def test_to_dict_serialization(monkeypatch):
    """
    Ensure to_dict uses correct field names and values, and nests a serialized document store.
    """
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="t",
        embedding_dim=768,
    )

    # Avoid depending on OracleDocumentStore.to_dict internals here.
    monkeypatch.setattr(
        store,
        "to_dict",
        lambda: {
            "type": "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore",
            "init_parameters": {
                "connection_params": connection_params,
                "table_name": "t",
                "embedding_dim": 768,
            },
        },
    )

    retriever = OracleEmbeddingRetriever(document_store=store)
    res = retriever.to_dict()

    assert res["type"] == (
        "haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever"
    )
    init = res["init_parameters"]
    assert init["filters"] == {}
    assert init["top_k"] == 10
    assert init["distance_strategy"] == "cosine"
    assert init["filter_policy"] == "replace"
    assert init["document_store"]["type"] == (
        "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore"
    )


def test_from_dict_roundtrip_and_default_filter_policy():
    """
    Ensure from_dict builds a retriever with a deserialized OracleDocumentStore and defaults filter_policy
    to REPLACE when missing.
    """
    connection_params = oracle_unit_test_connection_params()
    base_doc_store_dict = {
        "type": "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore",
        "init_parameters": {
            "connection_params": connection_params,
            "table_name": "t",
            "embedding_dim": 768,
        },
    }

    # With explicit filter_policy
    data_with_policy = {
        "type": "haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever",
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "distance_strategy": "cosine",
            "filter_policy": "replace",
            "document_store": base_doc_store_dict,
        },
    }
    r1 = OracleEmbeddingRetriever.from_dict(data_with_policy)
    assert isinstance(r1.document_store, OracleDocumentStore)
    assert r1.distance_strategy == "cosine"
    assert r1.filter_policy == FilterPolicy.REPLACE
    assert r1.top_k == 10

    # Without filter_policy -> default to REPLACE
    data_no_policy = {
        "type": "haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever",
        "init_parameters": {
            "filters": {},
            "top_k": 5,
            "distance_strategy": "dot",
            "document_store": base_doc_store_dict,
        },
    }
    r2 = OracleEmbeddingRetriever.from_dict(data_no_policy)
    assert isinstance(r2.document_store, OracleDocumentStore)
    assert r2.distance_strategy == "dot"
    assert r2.filter_policy == FilterPolicy.REPLACE
    assert r2.top_k == 5


def test_retriever_roundtrip_preserves_real_document_store():
    connection_params = oracle_unit_test_connection_params()
    retriever = OracleEmbeddingRetriever(
        document_store=OracleDocumentStore(
            connection_params=connection_params,
            table_name="docs",
            embedding_dim=768,
            create_vector_index=True,
            vector_index_params={"idx_name": "my_idx", "idx_type": "HNSW"},
            vector_index_distance_strategy="dot",
        ),
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=7,
        distance_strategy="euclidean",
        filter_policy=FilterPolicy.MERGE,
    )

    restored = OracleEmbeddingRetriever.from_dict(retriever.to_dict())

    assert restored.filters == {"field": "meta.topic", "operator": "==", "value": "science"}
    assert restored.top_k == 7
    assert restored.distance_strategy == "euclidean"
    assert restored.filter_policy == FilterPolicy.MERGE
    assert isinstance(restored.document_store, OracleDocumentStore)
    assert restored.document_store._table_name == '"docs"'
    assert restored.document_store._embedding_dim == 768
    assert restored.document_store._create_vector_index is True
    assert restored.document_store._vector_index_params == {"idx_name": "my_idx", "idx_type": "HNSW"}
    assert restored.document_store._vector_index_embedding_field == "embedding"
    assert restored.document_store._vector_index_distance_strategy == "dot"


@pytest.mark.asyncio
async def test_run_async_delegates_to_document_store():
    """
    Unit test: ensure run_async() delegates to the document store with expected arguments.
    """
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval_async = AsyncMock(return_value=[])

    retriever = OracleEmbeddingRetriever(document_store=mock_store)
    filters = {"field": "meta.topic", "operator": "==", "value": "foo"}
    res = await retriever.run_async(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters=filters,
        top_k=5,
        distance_strategy="cosine",
    )

    mock_store._embedding_retrieval_async.assert_awaited_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters=filters,
        top_k=5,
        distance_strategy="cosine",
    )
    assert "documents" in res
    assert isinstance(res["documents"], list)


@pytest.mark.asyncio
async def test_run_async_uses_instance_distance_strategy_when_not_overridden():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval_async = AsyncMock(return_value=[])

    retriever = OracleEmbeddingRetriever(document_store=mock_store, distance_strategy="euclidean")
    await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4])

    mock_store._embedding_retrieval_async.assert_awaited_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters={},
        top_k=10,
        distance_strategy="euclidean",
    )


def test_run_rejects_invalid_distance_override():
    mock_store = Mock(spec=OracleDocumentStore)
    retriever = OracleEmbeddingRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="Invalid distance_function"):
        retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], distance_strategy="bad")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_async_rejects_invalid_distance_override():
    mock_store = Mock(spec=OracleDocumentStore)
    retriever = OracleEmbeddingRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="Invalid distance_function"):
        await retriever.run_async(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            distance_strategy="bad",  # type: ignore[arg-type]
        )

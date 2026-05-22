# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import oracledb
import pytest
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.retrievers.oracle import (
    OracleSparseEmbeddingRetriever,
)

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
def sparse_document_store():
    """
    Create a fresh OracleDocumentStore configured for sparse embeddings
    and purge the table before/after.
    """
    connection_params = oracle_test_connection_params()
    connection = oracledb.connect(**connection_params)
    table_name = "mytable_haystack_sparse"

    drop_table_purge(connection, table_name)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name=table_name,
        embedding_dim=8,
        support_sparse_embeddings=True,
        create_vector_index=False,
        vector_index_embedding_field="sparse_embedding",
    )

    yield store

    drop_table_purge(connection, table_name)
    connection.close()


def test_sparse_retriever_init_and_serialize(monkeypatch):
    """
    Basic init and to_dict sanity for OracleSparseEmbeddingRetriever.
    """
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="t",
        embedding_dim=8,
        support_sparse_embeddings=True,
        vector_index_embedding_field="sparse_embedding",
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
                "embedding_dim": 8,
                "vector_index_embedding_field": "sparse_embedding",
            },
        },
    )

    retriever = OracleSparseEmbeddingRetriever(document_store=store)
    res = retriever.to_dict()

    assert res["type"] == (
        "haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever"
    )
    init = res["init_parameters"]
    assert init["filters"] == {}
    assert init["top_k"] == 10
    assert init["distance_strategy"] == "cosine"
    assert init["filter_policy"] == "replace"
    assert init["document_store"]["type"] == (
        "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore"
    )

    with pytest.raises(ValueError, match="document_store must be an instance"):
        OracleSparseEmbeddingRetriever(document_store=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid distance_function"):
        OracleSparseEmbeddingRetriever(document_store=store, distance_strategy="bad")  # type: ignore[arg-type]


def test_sparse_from_dict_defaults_filter_policy_when_missing():
    connection_params = oracle_unit_test_connection_params()
    retriever = OracleSparseEmbeddingRetriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore",
                    "init_parameters": {
                        "connection_params": connection_params,
                        "table_name": "t",
                        "embedding_dim": 8,
                    },
                },
                "filters": {},
                "top_k": 3,
                "distance_strategy": "cosine",
            },
        }
    )

    assert retriever.filter_policy == FilterPolicy.REPLACE


def test_sparse_run_uses_instance_distance_strategy_when_not_overridden():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval.return_value = []

    retriever = OracleSparseEmbeddingRetriever(document_store=mock_store, distance_strategy="dot")
    retriever.run(query_sparse_embedding=SparseEmbedding(indices=[0], values=[1.0]))

    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=SparseEmbedding(indices=[0], values=[1.0]),
        filters={},
        top_k=10,
        distance_strategy="dot",
    )


@pytest.mark.asyncio
async def test_sparse_run_async_uses_instance_distance_strategy_when_not_overridden():
    mock_store = Mock(spec=OracleDocumentStore)
    mock_store._embedding_retrieval_async = AsyncMock(return_value=[])

    retriever = OracleSparseEmbeddingRetriever(document_store=mock_store, distance_strategy="euclidean")
    await retriever.run_async(query_sparse_embedding=SparseEmbedding(indices=[0], values=[1.0]))

    mock_store._embedding_retrieval_async.assert_awaited_once_with(
        query_embedding=SparseEmbedding(indices=[0], values=[1.0]),
        filters={},
        top_k=10,
        distance_strategy="euclidean",
    )


def test_sparse_run_rejects_invalid_distance_override():
    mock_store = Mock(spec=OracleDocumentStore)
    retriever = OracleSparseEmbeddingRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="Invalid distance_function"):
        retriever.run(
            query_sparse_embedding=SparseEmbedding(indices=[0], values=[1.0]),
            distance_strategy="bad",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_sparse_run_async_rejects_invalid_distance_override():
    mock_store = Mock(spec=OracleDocumentStore)
    retriever = OracleSparseEmbeddingRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="Invalid distance_function"):
        await retriever.run_async(
            query_sparse_embedding=SparseEmbedding(indices=[0], values=[1.0]),
            distance_strategy="bad",  # type: ignore[arg-type]
        )


def test_sparse_retriever_roundtrip_preserves_real_document_store():
    connection_params = oracle_unit_test_connection_params()
    retriever = OracleSparseEmbeddingRetriever(
        document_store=OracleDocumentStore(
            connection_params=connection_params,
            table_name="docs",
            embedding_dim=8,
            support_sparse_embeddings=True,
            create_vector_index=True,
            vector_index_params={"idx_name": "my_sparse_idx", "idx_type": "HNSW"},
            vector_index_embedding_field="sparse_embedding",
            vector_index_distance_strategy="dot",
        ),
        filters={"field": "meta.topic", "operator": "==", "value": "science"},
        top_k=4,
        distance_strategy="euclidean",
        filter_policy=FilterPolicy.MERGE,
    )

    restored = OracleSparseEmbeddingRetriever.from_dict(retriever.to_dict())

    assert restored.filters == {"field": "meta.topic", "operator": "==", "value": "science"}
    assert restored.top_k == 4
    assert restored.distance_strategy == "euclidean"
    assert restored.filter_policy == FilterPolicy.MERGE
    assert isinstance(restored.document_store, OracleDocumentStore)
    assert restored.document_store._table_name == '"docs"'
    assert restored.document_store._embedding_dim == 8
    assert restored.document_store._create_vector_index is True
    assert restored.document_store._vector_index_params == {"idx_name": "my_sparse_idx", "idx_type": "HNSW"}
    assert restored.document_store._vector_index_embedding_field == "sparse_embedding"
    assert restored.document_store._vector_index_distance_strategy == "dot"


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.integration
def test_sparse_retrieval_sync(sparse_document_store: OracleDocumentStore):
    """
    Integration test for sparse retrieval (sync).
    """
    docs = [
        Document(id="1", content="first"),
        Document(id="2", content="second"),
        Document(id="3", content="third"),
    ]

    # Assign sparse embeddings
    docs[0].sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])
    docs[1].sparse_embedding = SparseEmbedding(indices=[0], values=[0.9])
    docs[2].sparse_embedding = SparseEmbedding(indices=[1], values=[1.0])

    num_written = sparse_document_store.write_documents(docs)
    assert num_written == 3

    retriever = OracleSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=2)

    query_sparse = SparseEmbedding(indices=[0], values=[1.0])
    res = retriever.run(query_sparse_embedding=query_sparse)
    out_docs = res["documents"]
    assert len(out_docs) == 2
    assert sorted([d.id for d in out_docs]) == ["1", "2"]


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_sparse_retrieval_async(sparse_document_store: OracleDocumentStore):
    """
    Integration test for sparse retrieval (async).
    """
    docs = [
        Document(id="a", content="alpha"),
        Document(id="b", content="beta"),
        Document(id="c", content="gamma"),
    ]

    docs[0].sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])
    docs[1].sparse_embedding = SparseEmbedding(indices=[0], values=[0.9])
    docs[2].sparse_embedding = SparseEmbedding(indices=[1], values=[1.0])

    num_written = sparse_document_store.write_documents(docs)
    assert num_written == 3

    retriever = OracleSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=2)

    query_sparse = SparseEmbedding(indices=[0], values=[1.0])
    res = await retriever.run_async(query_sparse_embedding=query_sparse)
    out_docs = res["documents"]
    assert len(out_docs) == 2
    assert sorted([d.id for d in out_docs]) == ["a", "b"]


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.integration
def test_sparse_preferred_when_dense_present(sparse_document_store: OracleDocumentStore):
    """
    When documents have both dense and sparse embeddings, a SparseEmbedding query must
    use the sparse embeddings for similarity and NOT the dense ones.
    """
    # Two documents with both embeddings:
    # - sparse_best: sparse matches the query; dense is not similar
    # - dense_best: dense is extremely similar to the query; sparse does not match
    docs = [
        Document(id="sparse_best", content="sparse_best"),
        Document(id="dense_best", content="dense_best"),
    ]

    # Assign sparse embeddings: make 'sparse_best' the closest to the sparse query
    docs[0].sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])  # exact match with query
    docs[1].sparse_embedding = SparseEmbedding(indices=[1], values=[1.0])  # different index

    # Assign dense embeddings: make 'dense_best' dense-closest to a one-hot [1,0,...] vector
    # while 'sparse_best' dense is not similar, to catch any accidental dense-based retrieval
    dim = 8
    docs[0].embedding = [0.0] * dim
    docs[1].embedding = [1.0] + [0.0] * (dim - 1)

    num_written = sparse_document_store.write_documents(docs)
    assert num_written == 2

    retriever = OracleSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=1)
    query_sparse = SparseEmbedding(indices=[0], values=[1.0])

    res = retriever.run(query_sparse_embedding=query_sparse)
    assert res["documents"][0].id == "sparse_best"

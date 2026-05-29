# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import uuid

import oracledb
import pytest
from haystack.dataclasses import SparseEmbedding
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
)
from haystack.utils import Secret

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.document_stores.oracle.document_store import _get_connection, _index_exists

from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connection_params,
    oracle_unit_test_connection_params,
)

EMBEDDING_DIM = 768


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, *_args, **_kwargs):
        self.query = query

    def fetchone(self):
        return (3,)


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def cursor(self):
        return self.cursor_obj


class _FakePooledConnection:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        self.released["value"] = True


class _FakePool:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    def acquire(self):
        return _FakePooledConnection(self.connection, self.released)


def drop_table_purge(connection: oracledb.Connection, table_name: str) -> None:
    ddl = f'DROP TABLE IF EXISTS "{table_name}" PURGE'

    with connection.cursor() as cursor:
        cursor.execute(ddl)


@pytest.fixture  # type: ignore
def document_store(request):
    """Base configuration."""
    connection_params = oracle_test_connection_params()

    connection = oracledb.connect(**connection_params)

    table_name = f"mytable_haystack_{uuid.uuid4().hex[:8]}"
    index_name = f"myindex_haystack_{uuid.uuid4().hex[:8]}"

    drop_table_purge(connection, table_name)

    store_type = request.param

    config = None

    if store_type == "default":
        config = OracleDocumentStore(
            connection_params=connection_params,
            table_name=table_name,
            embedding_dim=768,
            support_sparse_embeddings=False,
        )
    elif store_type == "index":
        config = OracleDocumentStore(
            connection_params=connection_params,
            table_name=table_name,
            embedding_dim=768,
            support_sparse_embeddings=False,
            create_vector_index=True,
            vector_index_params={"idx_name": index_name, "idx_type": "HNSW"},
        )
    elif store_type == "index_sparse":
        config = OracleDocumentStore(
            connection_params=connection_params,
            table_name=table_name,
            embedding_dim=768,
            create_vector_index=True,
            vector_index_params={"idx_name": index_name, "idx_type": "HNSW"},
            vector_index_embedding_field="sparse_embedding",
        )
    elif store_type == "pool":
        config = OracleDocumentStore(
            use_connection_pool=True,
            connection_params=connection_params,
            table_name=table_name,
            embedding_dim=768,
            support_sparse_embeddings=False,
            create_vector_index=True,
            vector_index_params={"idx_name": index_name, "idx_type": "HNSW"},
        )

    yield config

    drop_table_purge(connection, table_name)
    connection.close()


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.parametrize("document_store", ["default", "index", "index_sparse", "pool"], indirect=True)
@pytest.mark.integration
class TestDocumentStore(FilterDocumentsTest, CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    def test_write_documents(self, document_store: OracleDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_blob(self, document_store: OracleDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        document_store.write_documents(docs)

        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal.

        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method. This can happen for example when the Document Store sets
        a score to returned Documents. Since we can't know what the score will be, we can't compare
        the Documents reliably.
        """
        assert {x.id for x in received} == {x.id for x in expected}

    # ISO filter not supported.
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        pass

    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        pass

    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        pass

    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        pass


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.parametrize("document_store", ["default", "index", "index_sparse", "pool"], indirect=True)
@pytest.mark.integration
def test_write_documents(document_store: OracleDocumentStore):
    documents = [
        Document(id="1", content="Hello, world!", embedding=[0.1] * 768),
        Document(id="2", content="Hello, mum!", embedding=[0.3] * 768),
        Document(id="3", content="Hello, dad!", embedding=[0.2] * 768),
    ]
    document_store.write_documents(documents)

    retrieved_docs = document_store.filter_documents()
    retrieved_docs.sort(key=lambda x: x.id)

    for original_doc, retrieved_doc in zip(documents, retrieved_docs):
        assert original_doc.id == retrieved_doc.id
        assert original_doc.content == retrieved_doc.content
        assert len(original_doc.embedding) == len(retrieved_doc.embedding)
        # these embeddings are in half precision, so we increase the tolerance
        assert original_doc.embedding == pytest.approx(retrieved_doc.embedding, abs=5e-5)


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.parametrize("document_store", ["index", "index_sparse", "pool"], indirect=True)
@pytest.mark.integration
def test_index_exists(document_store: OracleDocumentStore):
    docs = [Document(id="1")]
    assert document_store.write_documents(docs) == 1

    connection = oracledb.connect(**oracle_test_connection_params())

    assert _index_exists(
        connection,
        table_name=f'"{document_store._table_name}"',
        index_name=f'"{document_store._vector_index_params["idx_name"]}"',
    )


def test_get_connection_releases_sync_pooled_connection(monkeypatch):
    released = {"value": False}
    pool = _FakePool(_FakeConnection(), released)

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    with _get_connection(pool) as connection:
        assert connection is pool.connection
        assert released["value"] is False

    assert released["value"] is True


def test_count_documents_releases_sync_pooled_connection(monkeypatch):
    released = {"value": False}
    pool = _FakePool(_FakeConnection(), released)
    connection_params = oracle_unit_test_connection_params()

    monkeypatch.setattr(oracledb, "ConnectionPool", _FakePool, raising=False)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        use_connection_pool=True,
    )
    store._client = pool
    store._initialized = True

    assert store.count_documents() == 3
    assert released["value"] is True


def test_write_documents_with_sparse_embedding_requires_embedding_dim():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=None,
    )
    store._initialized = True

    doc = Document(id="1", content="hello")
    doc.sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])

    with pytest.raises(ValueError, match="embedding_dim must be set"):
        store.write_documents([doc])


def test_sparse_embedding_retrieval_requires_embedding_dim():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=None,
    )
    store._initialized = True

    with pytest.raises(ValueError, match="embedding_dim must be set"):
        store._embedding_retrieval(SparseEmbedding(indices=[0], values=[1.0]))


def test_write_documents_with_sparse_embedding_requires_sparse_support():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    store._initialized = True

    doc = Document(id="1", content="hello")
    doc.sparse_embedding = SparseEmbedding(indices=[0], values=[1.0])

    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        store.write_documents([doc])


def test_sparse_embedding_retrieval_requires_sparse_support():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    store._initialized = True

    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        store._embedding_retrieval(SparseEmbedding(indices=[0], values=[1.0]))


def test_sparse_vector_index_enabled_uses_defaults():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=768,
        sparse_vector_index={"enabled": True},
    )

    assert store._sparse_vector_index == {
        "enabled": True,
        "distance_strategy": "cosine",
        "params": None,
    }


def test_sparse_vector_index_requires_sparse_support():
    connection_params = oracle_unit_test_connection_params()

    with pytest.raises(ValueError, match="support_sparse_embeddings=True"):
        OracleDocumentStore(
            connection_params=connection_params,
            table_name="docs",
            embedding_dim=768,
            support_sparse_embeddings=False,
            sparse_vector_index={"enabled": True},
        )


def test_document_store_to_dict_serializes_index_config():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        use_connection_pool=True,
        embedding_dim=768,
        support_sparse_embeddings=True,
        create_vector_index=True,
        vector_index_params={"idx_name": "my_idx", "idx_type": "HNSW"},
        vector_index_distance_strategy="dot",
        sparse_vector_index={
            "enabled": True,
            "distance_strategy": "dot",
            "params": {"idx_name": "my_sparse_idx", "idx_type": "HNSW"},
        },
    )

    assert store.to_dict() == {
        "type": "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore",
        "init_parameters": {
            "connection_params": {"user": None, "password": None, "dsn": None},
            "table_name": '"docs"',
            "use_connection_pool": True,
            "embedding_dim": 768,
            "support_sparse_embeddings": True,
            "create_vector_index": True,
            "vector_index_params": {"idx_name": "my_idx", "idx_type": "HNSW"},
            "vector_index_embedding_field": "embedding",
            "vector_index_distance_strategy": "dot",
            "sparse_vector_index": {
                "enabled": True,
                "distance_strategy": "dot",
                "params": {"idx_name": "my_sparse_idx", "idx_type": "HNSW"},
            },
        },
    }


def test_document_store_to_dict_serializes_secret_connection_params():
    store = OracleDocumentStore(
        connection_params={
            "user": Secret.from_env_var("ORACLE_USER"),
            "password": Secret.from_env_var("ORACLE_PASSWORD"),
            "dsn": Secret.from_env_var("ORACLE_DSN"),
            "wallet_password": "wallet-secret",
            "wallet_location": "/path/to/wallet",
            "events": True,
        },
        table_name="docs",
        embedding_dim=768,
    )

    serialized = store.to_dict()["init_parameters"]["connection_params"]
    assert serialized == {
        "user": {"type": "env_var", "env_vars": ["ORACLE_USER"], "strict": True},
        "password": {"type": "env_var", "env_vars": ["ORACLE_PASSWORD"], "strict": True},
        "dsn": {"type": "env_var", "env_vars": ["ORACLE_DSN"], "strict": True},
        "wallet_password": None,
        "wallet_location": None,
        "events": True,
    }


def test_document_store_to_dict_omits_non_string_sensitive_connection_params():
    store = OracleDocumentStore(
        connection_params={
            "access_token": ("token-value", "private-key-value"),
            "events": True,
        },
        table_name="docs",
        embedding_dim=768,
    )

    serialized = store.to_dict()["init_parameters"]["connection_params"]
    assert serialized == {
        "access_token": None,
        "events": True,
    }


def test_document_store_from_dict_roundtrip_preserves_index_config():
    connection_params = oracle_unit_test_connection_params()
    store = OracleDocumentStore.from_dict(
        {
            "type": "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore",
            "init_parameters": {
                "connection_params": connection_params,
                "table_name": "docs",
                "use_connection_pool": True,
                "embedding_dim": 768,
                "support_sparse_embeddings": True,
                "create_vector_index": True,
                "vector_index_params": {"idx_name": "my_idx", "idx_type": "HNSW"},
                "vector_index_embedding_field": "embedding",
                "vector_index_distance_strategy": "dot",
                "sparse_vector_index": {
                    "enabled": True,
                    "distance_strategy": "dot",
                    "params": {"idx_name": "my_sparse_idx", "idx_type": "HNSW"},
                },
            },
        }
    )

    assert store._connection_params == connection_params
    assert store._table_name == '"docs"'
    assert store._use_connection_pool is True
    assert store._embedding_dim == 768
    assert store._support_sparse_embeddings is True
    assert store._create_vector_index is True
    assert store._vector_index_params == {"idx_name": "my_idx", "idx_type": "HNSW"}
    assert store._vector_index_embedding_field == "embedding"
    assert store._vector_index_distance_strategy == "dot"
    assert store._distance_strategy == "dot"
    assert store._sparse_vector_index == {
        "enabled": True,
        "distance_strategy": "dot",
        "params": {"idx_name": "my_sparse_idx", "idx_type": "HNSW"},
    }


def test_document_store_from_dict_deserializes_secret_connection_params(monkeypatch):
    monkeypatch.setenv("ORACLE_USER", "onnxuser")
    monkeypatch.setenv("ORACLE_PASSWORD", "secret")
    monkeypatch.setenv("ORACLE_DSN", "database.example/pdb")

    store = OracleDocumentStore.from_dict(
        {
            "type": "haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore",
            "init_parameters": {
                "connection_params": {
                    "user": {"type": "env_var", "env_vars": ["ORACLE_USER"], "strict": True},
                    "password": {"type": "env_var", "env_vars": ["ORACLE_PASSWORD"], "strict": True},
                    "dsn": {"type": "env_var", "env_vars": ["ORACLE_DSN"], "strict": True},
                },
                "table_name": "docs",
                "embedding_dim": 768,
            },
        }
    )

    assert store._connection_params["user"].resolve_value() == "onnxuser"
    assert store._connection_params["password"].resolve_value() == "secret"
    assert store._connection_params["dsn"].resolve_value() == "database.example/pdb"

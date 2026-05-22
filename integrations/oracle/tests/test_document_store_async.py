# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import oracledb
import pytest
import pytest_asyncio
from haystack.dataclasses import SparseEmbedding
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore

from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connection_params,
    oracle_unit_test_connection_params,
)


class _AsyncFakeCursor:
    async def execute(self, query, *_args, **_kwargs):
        self.query = query

    async def fetchone(self):
        return (3,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _AsyncFakeConnection:
    def __init__(self):
        self.cursor_obj = _AsyncFakeCursor()

    def cursor(self):
        return self.cursor_obj


class _AsyncFakePooledConnection:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        self.released["value"] = True


class _AsyncFakePool:
    def __init__(self, connection, released):
        self.connection = connection
        self.released = released

    def acquire(self):
        return _AsyncFakePooledConnection(self.connection, self.released)


async def drop_table_purge_async(connection: oracledb.Connection, table_name: str) -> None:
    ddl = f'DROP TABLE IF EXISTS "{table_name}" PURGE'

    await connection.execute(ddl)


@pytest_asyncio.fixture  # type: ignore
async def document_store():
    """Base configuration."""
    connection_params = oracle_test_connection_params()

    connection = await oracledb.connect_async(**connection_params)

    await drop_table_purge_async(connection, "mytable_haystack")

    config = OracleDocumentStore(
        connection_params=connection_params,
        table_name="mytable_haystack",
        embedding_dim=2500,
        support_sparse_embeddings=False,
    )

    yield config

    await drop_table_purge_async(connection, "mytable_haystack")
    await connection.close()


@pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentStoreAsync:
    async def test_write_documents(self, document_store: OracleDocumentStore):
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    async def test_write_blob(self, document_store: OracleDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        await document_store.write_documents_async(docs)

        retrieved_docs = await document_store.filter_documents_async()
        assert retrieved_docs == docs

    async def test_count_documents(self, document_store: OracleDocumentStore):
        await document_store.write_documents_async(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    async def test_filter_documents(self, document_store: OracleDocumentStore):
        filterable_docs = [
            Document(
                content="1",
                meta={
                    "number": -10,
                },
            ),
            Document(
                content="2",
                meta={
                    "number": 100,
                },
            ),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )

        assert result == [d for d in filterable_docs if d.meta.get("number") == 100]

    async def test_delete_documents(self, document_store: OracleDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0


@pytest.mark.asyncio
async def test_count_documents_async_releases_pooled_connection(monkeypatch):
    released = {"value": False}
    pool = _AsyncFakePool(_AsyncFakeConnection(), released)
    connection_params = oracle_unit_test_connection_params()

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        use_connection_pool=True,
    )
    store._client_async = pool
    store._initialized_async = True

    assert await store.count_documents_async() == 3
    assert released["value"] is True


@pytest.mark.asyncio
async def test_sparse_embedding_retrieval_async_requires_embedding_dim(monkeypatch):
    released = {"value": False}
    pool = _AsyncFakePool(_AsyncFakeConnection(), released)
    connection_params = oracle_unit_test_connection_params()

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=None,
    )
    store._client_async = pool
    store._initialized_async = True

    with pytest.raises(ValueError, match="embedding_dim must be set"):
        await store._embedding_retrieval_async(SparseEmbedding(indices=[0], values=[1.0]))
    assert released["value"] is True


@pytest.mark.asyncio
async def test_sparse_embedding_retrieval_async_requires_sparse_support(monkeypatch):
    released = {"value": False}
    pool = _AsyncFakePool(_AsyncFakeConnection(), released)
    connection_params = oracle_unit_test_connection_params()

    monkeypatch.setattr(oracledb, "AsyncConnectionPool", _AsyncFakePool, raising=False)

    store = OracleDocumentStore(
        connection_params=connection_params,
        table_name="docs",
        embedding_dim=4,
        support_sparse_embeddings=False,
    )
    store._client_async = pool
    store._initialized_async = True

    with pytest.raises(ValueError, match="Sparse embeddings are not supported"):
        await store._embedding_retrieval_async(SparseEmbedding(indices=[0], values=[1.0]))
    assert released["value"] is True

# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, Mock

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.oracle import OracleDocumentEmbedder
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
    "meta_fields_to_embed": [],
    "embedding_separator": "\n",
}


class TestOracleTextEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for OracleDocumentEmbedder.
        """
        OracleDocumentEmbedder(**default_params)

    def test_to_dict(self):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        embedder_component = OracleDocumentEmbedder(**default_params)
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder",
            "init_parameters": {**default_params, "connection_params": {"user": None, "password": None, "dsn": None}},
        }

    def test_to_dict_serializes_secret_connection_params_and_proxy(self):
        embedder_component = OracleDocumentEmbedder(
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

    def test_from_dict(self):
        component_dict = {
            "type": "haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder",
            "init_parameters": {**default_params},
        }

        OracleDocumentEmbedder.from_dict(component_dict)

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embedding.
        """
        embedder = OracleDocumentEmbedder(**default_params)
        list_integers_input = ["text_snippet_1", "text_snippet_2"]

        with pytest.raises(TypeError):
            embedder.run(list_integers_input)  # type: ignore[arg-type]

    def test_prepare_texts_to_embed_uses_meta_and_empty_content(self):
        embedder = OracleDocumentEmbedder(
            **{
                **default_params,
                "meta_fields_to_embed": ["title", "page", "missing", "none_value"],
                "embedding_separator": " | ",
            }
        )

        documents = [
            Document(content="body", meta={"title": "Guide", "page": 3, "none_value": None}),
            Document(content=None, meta={"title": "Empty"}),
        ]

        assert embedder._prepare_texts_to_embed(documents) == [
            "Guide | 3 | body",
            "Empty | ",
        ]

    def test_run_sets_embeddings_on_documents(self, monkeypatch):
        embedder = OracleDocumentEmbedder(**default_params)
        documents = [Document(content="body"), Document(content="other")]
        embed_documents = Mock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        monkeypatch.setattr(embedder, "_embed_documents", embed_documents)

        result = embedder.run(documents)

        embed_documents.assert_called_once_with(["body", "other"])
        assert result["documents"] == documents
        assert documents[0].embedding == [0.1, 0.2]
        assert documents[1].embedding == [0.3, 0.4]
        assert result["meta"] == default_params["embedding_params"]

    @pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
    @pytest.mark.integration
    def test_run(self):
        embedder = OracleDocumentEmbedder(
            **{
                **default_params,
                "connection_params": {"dsn": oracle_test_connect_dsn()},
            }
        )

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = embedder.run(docs)

        assert result["meta"] == default_params["embedding_params"]

        for doc, doc_with_embedding in zip(docs, result["documents"]):
            assert doc_with_embedding.content == doc.content
            assert doc_with_embedding.meta == doc.meta
            assert len(doc_with_embedding.embedding) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ORACLE_TESTS_CONFIGURED, reason=ORACLE_TESTS_REASON)
    @pytest.mark.integration
    async def test_run_async(self):
        embedder = OracleDocumentEmbedder(
            **{
                **default_params,
                "connection_params": {"dsn": oracle_test_connect_dsn()},
            }
        )
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = await embedder.run_async(docs)

        assert result["meta"] == default_params["embedding_params"]

        for doc, doc_with_embedding in zip(docs, result["documents"]):
            assert doc_with_embedding.content == doc.content
            assert doc_with_embedding.meta == doc.meta
            assert len(doc_with_embedding.embedding) > 0

    @pytest.mark.asyncio
    async def test_run_async_sets_embeddings_on_documents(self, monkeypatch):
        embedder = OracleDocumentEmbedder(**default_params)
        documents = [Document(content="body"), Document(content="other")]
        embed_documents_async = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        monkeypatch.setattr(embedder, "_embed_documents_async", embed_documents_async)

        result = await embedder.run_async(documents)

        embed_documents_async.assert_awaited_once_with(["body", "other"])
        assert result["documents"] == documents
        assert documents[0].embedding == [0.1, 0.2]
        assert documents[1].embedding == [0.3, 0.4]
        assert result["meta"] == default_params["embedding_params"]

    @pytest.mark.asyncio
    async def test_run_async_wrong_input_format(self):
        embedder = OracleDocumentEmbedder(**default_params)

        with pytest.raises(TypeError):
            await embedder.run_async(["text_snippet_1"])  # type: ignore[arg-type]

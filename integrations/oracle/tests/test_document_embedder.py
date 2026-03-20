# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document

from haystack_integrations.components.embedders.oracle import OracleDocumentEmbedder
from .conftest import (
    ORACLE_TESTS_CONFIGURED,
    ORACLE_TESTS_REASON,
    oracle_test_connect_dsn,
)


default_params = {
    "connection_params": {"dsn": "user/password@host:1521/service_name"},
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
            "init_parameters": {**default_params},
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
            embedder.run(text=list_integers_input)

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

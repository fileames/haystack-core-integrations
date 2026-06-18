# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.oracle import OracleHybridRetriever
from haystack_integrations.document_stores.oracle import OracleDocumentStore


def test_search_params_for_hybrid_mode(patched_store):
    params = patched_store._hybrid_search_params(
        "oracle vector",
        index_name="TEST_HYBRID",
        search_mode="hybrid",
        top_k=5,
        params=None,
    )

    assert params["hybrid_index_name"] == "TEST_HYBRID"
    assert params["vector"]["search_text"] == "oracle vector"
    assert params["text"]["search_text"] == "oracle vector"
    assert params["return"]["topN"] == 5


def test_search_params_for_keyword_mode(patched_store):
    params = patched_store._hybrid_search_params(
        "oracle vector",
        index_name="TEST_HYBRID",
        search_mode="keyword",
        top_k=3,
        params=None,
    )

    assert "vector" not in params
    assert params["text"]["search_text"] == "oracle vector"
    assert params["return"]["topN"] == 3


def test_search_params_does_not_inject_filter_by(patched_store):
    # Haystack metadata filters are applied as a SQL post-filter, not pushed into
    # DBMS_HYBRID_VECTOR filter_by (whose paths resolve to base-table columns).
    params = patched_store._hybrid_search_params(
        "oracle",
        index_name="TEST_HYBRID",
        search_mode="hybrid",
        top_k=10,
        params=None,
    )

    assert "filter_by" not in params


def test_search_params_preserves_native_filter_by(patched_store):
    # A native filter_by over declared filterable columns is still honored when passed via params.
    native_filter = {"op": "=", "path": "category", "type": "string", "args": ["news"]}
    params = patched_store._hybrid_search_params(
        "oracle",
        index_name="TEST_HYBRID",
        search_mode="hybrid",
        top_k=10,
        params={"filter_by": native_filter},
    )

    assert params["filter_by"] == native_filter


def test_build_filter_fragment_produces_sql_predicate():
    fragment, params = OracleDocumentStore._build_filter_fragment(
        {"field": "meta.lang", "operator": "==", "value": "en"}
    )

    assert "JSON_VALUE(metadata, '$.lang')" in fragment
    assert params == {"p0": "en"}


def test_validate_params_rejects_derived_search_text(mock_store):
    with pytest.raises(ValueError, match="search_text"):
        OracleHybridRetriever(
            document_store=mock_store,
            index_name="TEST_HYBRID",
            params={"vector": {"search_text": "do not set"}},
        )


def test_filter_policy_string_is_supported(mock_store):
    retriever = OracleHybridRetriever(
        document_store=mock_store,
        index_name="TEST_HYBRID",
        filter_policy="merge",
    )

    assert retriever.filter_policy == FilterPolicy.MERGE


def test_run_calls_hybrid_retrieval(mock_store):
    documents = [Document(id="A" * 32, content="hi")]
    mock_store._hybrid_retrieval.return_value = documents
    retriever = OracleHybridRetriever(
        document_store=mock_store,
        index_name="TEST_HYBRID",
        search_mode="semantic",
        top_k=5,
        params={"vector": {"score_weight": 2}},
        return_scores=True,
    )
    filters = {"field": "meta.lang", "operator": "==", "value": "en"}

    result = retriever.run("oracle", filters=filters, top_k=3, params={"text": {"score_weight": 1}})

    assert result["documents"] is documents
    mock_store._hybrid_retrieval.assert_called_once_with(
        "oracle",
        index_name="TEST_HYBRID",
        search_mode="semantic",
        filters=filters,
        top_k=3,
        params={"vector": {"score_weight": 2}, "text": {"score_weight": 1}},
        return_scores=True,
    )


@pytest.mark.asyncio
async def test_run_async_calls_hybrid_retrieval_async(mock_store):
    retriever = OracleHybridRetriever(document_store=mock_store, index_name="TEST_HYBRID", top_k=5)

    result = await retriever.run_async("oracle")

    mock_store._hybrid_retrieval_async.assert_called_once_with(
        "oracle",
        index_name="TEST_HYBRID",
        search_mode="hybrid",
        filters={},
        top_k=5,
        params={},
        return_scores=False,
    )
    assert "documents" in result


def test_to_dict_from_dict_roundtrip(mock_store):
    retriever = OracleHybridRetriever(
        document_store=mock_store,
        index_name="TEST_HYBRID",
        search_mode="semantic",
        filters={"field": "meta.lang", "operator": "==", "value": "en"},
        top_k=2,
        return_scores=True,
    )

    data = retriever.to_dict()
    restored = OracleHybridRetriever.from_dict(data)

    assert restored.index_name == "TEST_HYBRID"
    assert restored.search_mode == "semantic"
    assert restored.filters == {"field": "meta.lang", "operator": "==", "value": "en"}
    assert restored.top_k == 2
    assert restored.return_scores is True

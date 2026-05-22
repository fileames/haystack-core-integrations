# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore


def _generate_accum_query(query: str, *, fuzzy: bool = False) -> str:
    tokens = [token for token in re.split(r"\W+", query) if token]
    if not tokens:
        raise ValueError("query must contain at least one searchable token.")

    if fuzzy:
        return " ACCUM ".join(f'fuzzy("{token}")' for token in tokens)
    return " ACCUM ".join(f'"{token}"' for token in tokens)


@component
class OracleTextRetriever:
    """
    Retrieve documents from an OracleDocumentStore using Oracle Text CONTAINS queries on the content column.
    """

    def __init__(
        self,
        document_store: OracleDocumentStore,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        fuzzy: bool = False,
        operator_search: bool = False,
        return_scores: bool = False,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        if not isinstance(document_store, OracleDocumentStore):
            raise ValueError("document_store must be an instance of OracleDocumentStore")

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.fuzzy = fuzzy
        self.operator_search = operator_search
        self.return_scores = return_scores
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    @staticmethod
    def _prepare_query(query: str, *, fuzzy: bool = False, operator_search: bool = False) -> str:
        if not isinstance(query, str):
            raise TypeError("OracleTextRetriever expects query to be a string.")

        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be empty.")

        if operator_search:
            return normalized
        return _generate_accum_query(normalized, fuzzy=fuzzy)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            fuzzy=self.fuzzy,
            operator_search=self.operator_search,
            return_scores=self.return_scores,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleTextRetriever":
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = OracleDocumentStore.from_dict(doc_store_params)
        filter_policy = data["init_parameters"].get("filter_policy")
        if filter_policy:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        else:
            data["init_parameters"]["filter_policy"] = FilterPolicy.REPLACE
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        fuzzy: Optional[bool] = None,
        operator_search: Optional[bool] = None,
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        fuzzy = self.fuzzy if fuzzy is None else fuzzy
        operator_search = self.operator_search if operator_search is None else operator_search
        prepared_query = self._prepare_query(query, fuzzy=fuzzy, operator_search=operator_search)

        documents = self.document_store._text_retrieval(query=prepared_query, filters=filters, top_k=top_k)
        if self.return_scores:
            for document in documents:
                if document.score is not None:
                    document.meta["score"] = document.score

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        fuzzy: Optional[bool] = None,
        operator_search: Optional[bool] = None,
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        fuzzy = self.fuzzy if fuzzy is None else fuzzy
        operator_search = self.operator_search if operator_search is None else operator_search
        prepared_query = self._prepare_query(query, fuzzy=fuzzy, operator_search=operator_search)

        documents = await self.document_store._text_retrieval_async(query=prepared_query, filters=filters, top_k=top_k)
        if self.return_scores:
            for document in documents:
                if document.score is not None:
                    document.meta["score"] = document.score

        return {"documents": documents}

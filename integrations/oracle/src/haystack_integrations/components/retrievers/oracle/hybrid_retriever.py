"""Oracle hybrid retriever component.

Executes DBMS_HYBRID_VECTOR.SEARCH against a hybrid vector index and returns
Haystack Documents from OracleDocumentStore. Supports keyword, semantic, and
hybrid modes plus Haystack-style metadata filters translated to Oracle
`filter_by` expressions.
"""

import inspect
import json
from typing import Any, Literal, Optional, Union

import oracledb
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from haystack.errors import FilterError

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.document_stores.oracle.document_store import (
    _get_connection,
    _get_connection_async,
    _quote_identifier,
    output_type_string_handler,
)
from haystack_integrations.components.document_stores.oracle.filters import _to_hybrid_filter

VALID_SEARCH_MODES = ("keyword", "hybrid", "semantic")


@component
class OracleHybridRetriever:
    """
    Retrieve documents from Oracle using DBMS_HYBRID_VECTOR.SEARCH.

    The retriever requires an existing hybrid vector index and can run in
    keyword-only, semantic-only, or combined hybrid mode.
    """

    def __init__(
        self,
        document_store: OracleDocumentStore,
        idx_name: str,
        *,
        search_mode: Literal["keyword", "hybrid", "semantic"] = "hybrid",
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        params: Optional[dict[str, Any]] = None,
        return_scores: bool = False,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        if not isinstance(document_store, OracleDocumentStore):
            raise ValueError("document_store must be an instance of OracleDocumentStore")
        if search_mode not in VALID_SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {VALID_SEARCH_MODES}.")

        self.document_store = document_store
        self.idx_name = _quote_identifier(idx_name)
        self.search_mode = search_mode
        self.filters = filters or {}
        self.top_k = top_k
        self.params = self._validate_params(params or {})
        self.return_scores = return_scores
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    @staticmethod
    def _validate_params(search_params: dict[str, Any]) -> dict[str, Any]:
        if "search_text" in search_params:
            raise ValueError(
                "Cannot provide search_text as a parameter at the top level; it is derived from the query."
            )
        if "return" in search_params:
            raise ValueError(
                "Cannot provide return as a parameter in params; it is handled internally. "
                "Use `return_scores` to include scores in the returned documents."
            )

        vector_params = search_params.get("vector") or {}
        if "search_text" in vector_params or "search_vector" in vector_params:
            raise ValueError(
                "Cannot provide search_text or search_vector in params['vector']; it is derived from the query."
            )

        text_params = search_params.get("text") or {}
        if (
            "search_text" in text_params
            or "search_vector" in text_params
            or "contains" in text_params
            or "json_textcontains" in text_params
        ):
            raise ValueError(
                "Cannot provide search_text, search_vector, contains, or json_textcontains in params['text']; "
                "they are derived from the query."
            )

        return search_params

    @staticmethod
    def _decode_search_result(value: Any) -> list[dict[str, Any]]:
        if hasattr(value, "read"):
            value = value.read()
        return json.loads(value)

    @staticmethod
    async def _decode_search_result_async(value: Any) -> list[dict[str, Any]]:
        if hasattr(value, "read"):
            value = value.read()
            if inspect.isawaitable(value):
                value = await value
        return json.loads(value)

    def _get_search_params(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        search_params = dict(self.params)
        search_params.update(self._validate_params(params or {}))
        search_params["hybrid_index_name"] = self.idx_name

        if self.search_mode in {"hybrid", "semantic"}:
            search_params["vector"] = dict(search_params.get("vector") or {})
            search_params["vector"]["search_text"] = query

        if self.search_mode in {"hybrid", "keyword"}:
            search_params["text"] = dict(search_params.get("text") or {})
            search_params["text"]["search_text"] = query

        if filters:
            if "filter_by" in search_params:
                raise FilterError("Cannot combine Haystack filters with params['filter_by']; provide only one.")
            search_params["filter_by"] = _to_hybrid_filter(filters)

        search_params["return"] = {
            "topN": top_k or self.top_k,
            "values": ["rowid", "score", "vector_score", "text_score"],
            "format": "JSON",
        }
        return search_params

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        search_params = self._get_search_params(query, filters=filters, top_k=top_k, params=params)
        self.document_store._ensure_initialized()

        with _get_connection(self.document_store._client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                cursor.execute(
                    "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                    search_params=search_params,
                )
                search_rows = self._decode_search_result(cursor.fetchone()[0])

                cursor.outputtypehandler = output_type_string_handler
                rows = []
                columns: list[str] | None = None
                for row in search_rows:
                    cursor.execute(
                        f"SELECT * FROM {self.document_store._table_name} WHERE rowid = :rid",
                        rid=row["rowid"],
                    )
                    if columns is None:
                        columns = [col.name for col in cursor.description]
                    rows.extend(cursor.fetchall())

        documents = self.document_store._get_result_to_documents(rows, columns or [])
        for row, document in zip(search_rows, documents, strict=False):
            document.score = row["score"]
            if self.return_scores:
                document.meta["score"] = row["score"]
                document.meta["text_score"] = row["text_score"]
                document.meta["vector_score"] = row["vector_score"]

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        search_params = self._get_search_params(query, filters=filters, top_k=top_k, params=params)
        await self.document_store._ensure_initialized_async()

        async with _get_connection_async(self.document_store._client_async) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                await cursor.execute(
                    "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                    search_params=search_params,
                )
                search_rows = await self._decode_search_result_async((await cursor.fetchone())[0])

                cursor.outputtypehandler = output_type_string_handler
                rows = []
                columns: list[str] | None = None
                for row in search_rows:
                    await cursor.execute(
                        f"SELECT * FROM {self.document_store._table_name} WHERE rowid = :rid",
                        rid=row["rowid"],
                    )
                    if columns is None:
                        columns = [col.name for col in cursor.description]
                    rows.extend(await cursor.fetchall())

        documents = self.document_store._get_result_to_documents(rows, columns or [])
        for row, document in zip(search_rows, documents, strict=False):
            document.score = row["score"]
            if self.return_scores:
                document.meta["score"] = row["score"]
                document.meta["text_score"] = row["text_score"]
                document.meta["vector_score"] = row["vector_score"]

        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            idx_name=self.idx_name,
            search_mode=self.search_mode,
            filters=self.filters,
            top_k=self.top_k,
            params=self.params,
            return_scores=self.return_scores,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleHybridRetriever":
        document_store = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = OracleDocumentStore.from_dict(document_store)

        filter_policy = data["init_parameters"].get("filter_policy")
        if filter_policy:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        else:
            data["init_parameters"]["filter_policy"] = FilterPolicy.REPLACE

        return default_from_dict(cls, data)

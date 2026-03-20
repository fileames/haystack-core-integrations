"""Oracle Sparse Embedding Retriever component.

Retrieves Documents from OracleDocumentStore using vector distance functions on sparse embeddings.
Provides synchronous and asynchronous interfaces, supports metadata filtering with FilterPolicy,
and configurable distance strategies ("dot", "euclidean", "cosine").
"""
from typing import Any, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore

VALID_DISTANCE_FUNCTIONS = "dot", "euclidean", "cosine"


@component
class OracleSparseEmbeddingRetriever:
    """
    Retrieve documents from an OracleDocumentStore based on sparse embedding similarity.

    This component delegates retrieval to OracleDocumentStore, which executes a vector
    similarity query in Oracle using the configured distance strategy. Runtime filters
    are merged with those defined at initialization using the selected FilterPolicy.
    """

    def __init__(
        self,
        document_store: OracleDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = "cosine",
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Initialize the OracleSparseEmbeddingRetriever.

        :param document_store: OracleDocumentStore instance used to execute vector similarity queries.
        :param filters: Optional base filters applied to every retrieval. Runtime filters provided to run/run_async
            are merged with these according to filter_policy.
        :param top_k: Maximum number of Documents to return.
        :param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
        :param filter_policy: Policy determining how runtime filters are merged with base filters.
        :raises ValueError: If document_store is not an OracleDocumentStore or if distance_strategy is invalid.
        """
        if not isinstance(document_store, OracleDocumentStore):
            msg = "document_store must be an instance of OracleDocumentStore"
            raise ValueError(msg)

        if distance_strategy not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_strategy}' for vector similarity. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.distance_strategy = distance_strategy
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            distance_strategy=self.distance_strategy,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleSparseEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
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
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the OracleDocumentStore based on a sparse query embedding.

        :param query_sparse_embedding: SparseEmbedding representing the query.
        :param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
        :param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
        :param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
            Defaults to the value set at initialization.
        :returns: A dictionary with:
            - documents: list of Documents similar to the given sparse embedding.
        :raises ValueError: If distance_strategy is invalid.
        """
        if distance_strategy is not None and distance_strategy not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_strategy}' for vector similarity. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        distance_strategy = distance_strategy or self.distance_strategy

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_sparse_embedding,
            filters=filters,
            top_k=top_k,
            distance_strategy=distance_strategy,
        )
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents from the OracleDocumentStore based on a sparse query embedding.

        :param query_sparse_embedding: SparseEmbedding representing the query.
        :param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
        :param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
        :param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
            Defaults to the value set at initialization.
        :returns: A dictionary with:
            - documents: list of Documents similar to the given sparse embedding.
        :raises ValueError: If distance_strategy is invalid.
        """
        if distance_strategy is not None and distance_strategy not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_strategy}' for vector similarity. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        distance_strategy = distance_strategy or self.distance_strategy

        docs = await self.document_store._embedding_retrieval_async(
            query_embedding=query_sparse_embedding,
            filters=filters,
            top_k=top_k,
            distance_strategy=distance_strategy,
        )
        return {"documents": docs}

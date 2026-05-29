"""Oracle Embedding Retriever component.

Retrieves Documents from OracleDocumentStore using vector distance functions on embeddings.
Provides synchronous and asynchronous interfaces, supports metadata filtering with
FilterPolicy, and configurable distance strategies ("dot", "euclidean", "cosine").
"""
from typing import Any, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.document_stores.oracle.document_store import _validate_top_k

VALID_DISTANCE_FUNCTIONS = "dot", "euclidean", "cosine"


@component
class OracleEmbeddingRetriever:
    """
    Retrieve documents from an OracleDocumentStore based on dense embedding similarity.

    This component delegates retrieval to OracleDocumentStore, which executes a vector
    similarity query in Oracle using the configured distance strategy. Runtime filters
    are merged with those defined at initialization using the selected FilterPolicy.

    Example:
    ```python
    import os
    from haystack import Document, Pipeline
    from haystack.document_stores.types import DuplicatePolicy

    from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
    from haystack_integrations.components.embedders.oracle import (
        OracleTextEmbedder,
        OracleDocumentEmbedder,
    )
    from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever

    # Create the document store (adjust connection params)
    store = OracleDocumentStore(
        connection_params={"dsn": os.environ["ORACLE_DB_DSN"]},
        table_name="documents",
        embedding_dim=768,
        create_vector_index=True,  # optional but recommended
        vector_index_distance_strategy="cosine",
    )

    # Prepare and write documents with embeddings
    docs = [
        Document(content="There are over 7,000 languages spoken around the world today."),
        Document(content="Elephants have been observed to behave in a way that indicates..."),
        Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
    ]

    doc_embedder = OracleDocumentEmbedder(
        connection_params={"dsn": os.environ["ORACLE_DB_DSN"]},
        embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
        proxy=None,
        use_connection_pool=False,
        meta_fields_to_embed=None,
    )
    docs_with_embeddings = doc_embedder.run(docs)["documents"]
    store.write_documents(docs_with_embeddings, policy=DuplicatePolicy.OVERWRITE)

    # Build a pipeline that embeds the query and retrieves similar documents
    pipe = Pipeline()
    pipe.add_component(
        "text_embedder",
        OracleTextEmbedder(
            connection_params={"dsn": os.environ["ORACLE_DB_DSN"]},
            embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
            proxy=None,
            use_connection_pool=False,
        ),
    )
    pipe.add_component("retriever", OracleEmbeddingRetriever(document_store=store, top_k=3))
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")

    res = pipe.run({"text_embedder": {"text": "How many languages are there?"}})
    assert "languages" in res["retriever"]["documents"][0].content
    ```
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
        Initialize the OracleEmbeddingRetriever.

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
        self.top_k = _validate_top_k(top_k)
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
    def from_dict(cls, data: dict[str, Any]) -> "OracleEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = OracleDocumentStore.from_dict(doc_store_params)
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        filter_policy = data["init_parameters"].get("filter_policy")
        if filter_policy:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        else:
            # default to REPLACE for backward compatibility
            data["init_parameters"]["filter_policy"] = FilterPolicy.REPLACE
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the OracleDocumentStore based on a query embedding.

        Runtime filters are merged with the retriever's base filters using the configured filter_policy.

        :param query_embedding: Embedding vector representing the query.
        :param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
        :param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
        :param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
            Defaults to the value set at initialization.
        :returns: A dictionary with:
            - documents: list of Documents similar to query_embedding.
        :raises ValueError: If distance_strategy is invalid.
        """
        if distance_strategy is not None and distance_strategy not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_strategy}' for vector similarity. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = _validate_top_k(self.top_k if top_k is None else top_k)
        distance_strategy = distance_strategy or self.distance_strategy

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            distance_strategy=distance_strategy,
        )
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents from the OracleDocumentStore based on a query embedding.

        Runtime filters are merged with the retriever's base filters using the configured filter_policy.

        :param query_embedding: Embedding vector representing the query.
        :param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
        :param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
        :param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
            Defaults to the value set at initialization.
        :returns: A dictionary with:
            - documents: list of Documents similar to query_embedding.
        :raises ValueError: If distance_strategy is invalid.
        """
        if distance_strategy is not None and distance_strategy not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_strategy}' for vector similarity. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = _validate_top_k(self.top_k if top_k is None else top_k)
        distance_strategy = distance_strategy or self.distance_strategy

        docs = await self.document_store._embedding_retrieval_async(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            distance_strategy=distance_strategy,
        )
        return {"documents": docs}

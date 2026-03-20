# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Oracle Document Embedder component.

This module provides OracleDocumentEmbedder, a Haystack component that computes vector embeddings
for lists of Haystack Documents using Oracle Database vector capabilities. It extends
OracleTextEmbedder by handling Document objects, optional inclusion of selected metadata fields,
and synchronous/asynchronous execution.
"""
from typing import Any, Optional

from haystack import Document, component, default_to_dict

from .text_embedder import OracleTextEmbedder


@component
class OracleDocumentEmbedder(OracleTextEmbedder):
    """
    Embed Haystack Documents with Oracle Database.

    This component concatenates selected metadata fields with the Document content and
    requests embeddings from Oracle Database. The resulting vectors are assigned back
    to the corresponding Document.embedding fields.
    """
    def __init__(
        self,
        connection_params: dict[str, Any],
        embedding_params: dict[str, Any],
        *,
        use_connection_pool: bool = False,
        proxy: Optional[str],
        meta_fields_to_embed: list[str] = [],
        embedding_separator: str = "\n",
    ):
        """
        Create an OracleDocumentEmbedder component.

        :param connection_params: Connection parameters for python-oracledb. Required.
            See the python-oracledb docs (https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html).
        :param embedding_params: Embedding parameters passed to Oracle embeddings (for example, provider, model, etc.).
            See the Oracle embedding docs (https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/utl_to_embedding-and-utl_to_embeddings-dbms_vector.html)
            for accepted values.
        :param use_connection_pool: If True, use a python-oracledb connection pool for connections. Defaults to False.
        :param proxy: Optional HTTP proxy to set via UTL_HTTP.set_proxy for outbound calls in the database session.
        :param meta_fields_to_embed: Optional list of keys from Document.meta whose values will be concatenated with the
            Document content before embedding. Keys missing in a Document or with None values are skipped.
            If None or empty, only the Document content is used.
        :param embedding_separator: String used to join selected metadata values and the Document content. Defaults to "\n".
        """  # noqa: E501
        super(OracleDocumentEmbedder, self).__init__(
            connection_params=connection_params,
            embedding_params=embedding_params,
            use_connection_pool=use_connection_pool,
            proxy=proxy,
        )
        self._meta_fields_to_embed = meta_fields_to_embed
        self._embedding_separator = embedding_separator

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Build per-document input strings for the Oracle embedding function.

        For each Document, this concatenates the values of the configured meta_fields_to_embed (if any)
        followed by the Document content, using embedding_separator as a delimiter. Metadata keys that
        are missing or whose values are None are skipped. If Document.content is None, an empty string
        is used.

        :param documents: List of Documents to convert to text inputs.
        :returns: A list of strings, one for each input Document, in the same order.
        """
        texts_to_embed: list[str] = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self._meta_fields_to_embed if doc.meta.get(key) is not None
            ]

            text_to_embed = self._embedding_separator.join(meta_values_to_embed + [doc.content or ""])  # noqa: RUF005
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute embeddings for a list of Documents.

        Each Document's embedding field is set in-place. The text passed to the Oracle embedding
        function is constructed from selected metadata fields and the Document content:

            "{meta_field_1}{separator}{meta_field_2}{separator}...{separator}{content}"

        Where the set of metadata fields comes from meta_fields_to_embed and the separator is embedding_separator.

        :param documents: List of Haystack Documents to embed. If a Document has no content, an empty string is used.
        :returns: A dictionary with:
            - documents: The same list of Documents with their embedding fields populated.
            - meta: The embedding parameters used for the call (for example, provider, model, etc.).
        :raises TypeError: If the input is not a list of Documents.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "OracleDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OracleTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents)
        embeddings = self._embed_documents(texts_to_embed)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": self._embedding_params}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously compute embeddings for a list of Documents.

        Behavior matches run(), but uses the async Oracle client.

        :param documents: List of Haystack Documents to embed. If a Document has no content, an empty string is used.
        :returns: A dictionary with:
            - documents: The same list of Documents with their embedding fields populated.
            - meta: The embedding parameters used for the call (for example, provider, model, etc.).
        :raises TypeError: If the input is not a list of Documents.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "OracleDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OracleTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents)
        embeddings = await self._embed_documents_async(texts_to_embed)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": self._embedding_params}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_params=self._connection_params,
            embedding_params=self._embedding_params,
            use_connection_pool=self._use_connection_pool,
            proxy=self._proxy,
            meta_fields_to_embed=self._meta_fields_to_embed,
            embedding_separator=self._embedding_separator,
        )

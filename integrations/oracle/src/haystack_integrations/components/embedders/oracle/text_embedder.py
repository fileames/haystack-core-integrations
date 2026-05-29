# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from collections.abc import Callable
from typing import Any

import oracledb
from haystack import component, default_from_dict, default_to_dict

from haystack_integrations.components.document_stores.oracle.document_store import (
    _compare_version,
    _deserialize_connection_params,
    _deserialize_optional_secret,
    _get_connection,
    _get_connection_async,
    _resolve_connection_params,
    _resolve_optional_secret,
    _serialize_connection_params,
    _serialize_optional_secret,
)

logger = logging.getLogger(__name__)


@component
class OracleTextEmbedder:
    """
    A component for embedding strings using Oracle Database.

    It connects to Oracle Database and retrieves embeddings for input text using the configured
    provider/model parameters.
    """

    def __init__(
        self,
        connection_params: dict[str, Any],
        embedding_params: dict[str, Any],
        *,
        use_connection_pool: bool = False,
        proxy: Any | None,
    ):
        """
        Creates a new OracleTextEmbedder component.

        :param connection_params: Connection parameters for python-oracledb. Required.
            See the python-oracledb docs (https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html).
            Values can be Haystack `Secret` instances to avoid serializing raw credentials.
        :param embedding_params: Embedding parameters passed to Oracle embeddings (for example, provider, model, etc.).
            See the Oracle embedding docs (https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/utl_to_embedding-and-utl_to_embeddings-dbms_vector.html)
            for accepted values.
        :param use_connection_pool: If True, use a python-oracledb connection pool for connections. Defaults to False.
        :param proxy: Optional HTTP proxy to set via UTL_HTTP.set_proxy for outbound calls in the database session.
            Can be a Haystack `Secret` to avoid serializing proxy credentials.
        """

        self._connection_params = connection_params
        self._embedding_params = embedding_params
        self._use_connection_pool = use_connection_pool
        self._proxy = proxy
        self._initialized = False
        self._initialized_async = False

        self._client = None
        self._client_async = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        connection_params = init_params.get("connection_params")
        if isinstance(connection_params, dict):
            _deserialize_connection_params(connection_params)
        _deserialize_optional_secret(init_params, "proxy")
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_params=self._serialized_connection_params(),
            embedding_params=self._embedding_params,
            use_connection_pool=self._use_connection_pool,
            proxy=self._serialized_proxy(),
        )

    def _serialized_connection_params(self) -> dict[str, Any]:
        return _serialize_connection_params(self._connection_params)

    def _serialized_proxy(self) -> dict[str, Any] | None:
        return _serialize_optional_secret(self._proxy)

    def _ensure_initialized(self):
        """
        Ensures the Oracle client is initialized and a connection or pool is created.

        Checks the installed python-oracledb version for vector support and initializes
        either a dedicated connection or a connection pool based on use_connection_pool.
        :raises Exception: If the python-oracledb version does not support vector operations (>= 2.2.0 required).
        """
        if self._initialized:
            return

        if _compare_version(oracledb.__version__, "2.2.0"):
            raise Exception(
                f"Oracle DB python client driver version {oracledb.__version__} not supported, \
                must be >=2.2.0 for vector support"
            )

        resolved_connection_params = _resolve_connection_params(self._connection_params)
        if self._use_connection_pool:
            self._client = oracledb.create_pool(**resolved_connection_params)
        else:
            self._client = oracledb.connect(**resolved_connection_params)

        self._initialized = True

    async def _ensure_initialized_async(self):
        """
        Ensures the async Oracle client is initialized and a connection or pool is created.

        Checks the installed python-oracledb version for vector support and initializes
        either a dedicated async connection or an async connection pool based on use_connection_pool.
        :raises Exception: If the python-oracledb version does not support vector operations (>= 2.2.0 required).
        """
        if self._initialized_async:
            return

        if _compare_version(oracledb.__version__, "2.2.0"):
            raise Exception(
                f"Oracle DB python client driver version {oracledb.__version__} not supported, \
                must be >=2.2.0 for vector support"
            )

        resolved_connection_params = _resolve_connection_params(self._connection_params)
        if self._use_connection_pool:
            self._client_async = await oracledb.create_pool_async(**resolved_connection_params)
        else:
            self._client_async = await oracledb.connect_async(**resolved_connection_params)

        self._initialized_async = True

    async def _handle_context(
        self,
        context: Callable[
            [oracledb.Connection | oracledb.AsyncConnection],
            Any,
        ],
    ) -> Any:
        """
        Helper to execute operations with a managed async connection.

        AsyncConnectionPool connections are not released automatically and must be used
        within a context manager. This method acquires a connection and ensures the
        appropriate context handling for pooled and non-pooled clients.

        :param context: A coroutine function that receives an async connection and performs the operation.
        :returns: The result returned by the provided context function.
        """
        async with _get_connection_async(self._client_async) as connection:
            return await context(connection)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings for a list of texts using Oracle Database.

        :param texts: The list of texts to embed.
        :returns: A list of embeddings, one for each input text.
        """

        self._ensure_initialized()

        embeddings: list[list[float]] = []
        # returns strings or bytes instead of a locator
        oracledb.defaults.fetch_lobs = False

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                proxy_was_set = False
                proxy = _resolve_optional_secret(self._proxy)
                if proxy:
                    cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=proxy)
                    proxy_was_set = True

                try:
                    chunks = []
                    for i, text in enumerate(texts, start=1):
                        chunk = {"chunk_id": i, "chunk_data": text}
                        chunks.append(json.dumps(chunk))

                    vector_array_type = connection.gettype("SYS.VECTOR_ARRAY_T")
                    inputs = vector_array_type.newobject(chunks)
                    cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
                    cursor.execute(
                        "select t.* from dbms_vector_chain.utl_to_embeddings(:1, json(:2)) t",
                        [inputs, self._embedding_params],
                    )

                    for row in cursor:
                        if row is None:
                            embeddings.append([])
                        else:
                            rdata = json.loads(row[0])
                            # dereference string as array
                            vec = json.loads(rdata["embed_vector"])
                            embeddings.append(vec)
                except BaseException as embedding_error:
                    if proxy_was_set:
                        try:
                            cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=None)
                        except Exception as cleanup_error:
                            msg = "Failed to clear Oracle session proxy after embedding failed"
                            logger.exception("%s; original embedding error was: %r", msg, embedding_error)
                            raise RuntimeError(msg) from cleanup_error
                    raise
                else:
                    if proxy_was_set:
                        try:
                            cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=None)
                        except Exception as cleanup_error:
                            msg = "Failed to clear Oracle session proxy after embedding succeeded"
                            logger.exception(msg)
                            raise RuntimeError(msg) from cleanup_error

        return embeddings

    async def _embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously compute embeddings for a list of texts using Oracle Database.

        :param texts: The list of texts to embed.
        :returns: A list of embeddings, one for each input text.
        """

        await self._ensure_initialized_async()

        async def context(connection: oracledb.AsyncConnection) -> None:
            embeddings: list[list[float]] = []

            oracledb.defaults.fetch_lobs = False

            with connection.cursor() as cursor:
                proxy_was_set = False
                proxy = _resolve_optional_secret(self._proxy)
                if proxy:
                    await cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=proxy)
                    proxy_was_set = True

                try:
                    chunks = []
                    for i, text in enumerate(texts, start=1):
                        chunk = {"chunk_id": i, "chunk_data": text}
                        chunks.append(json.dumps(chunk))

                    vector_array_type = await connection.gettype("SYS.VECTOR_ARRAY_T")
                    inputs = vector_array_type.newobject()
                    for v in chunks:
                        clob = await connection.createlob(oracledb.DB_TYPE_CLOB)
                        await clob.write(v)
                        inputs.append(clob)

                    cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
                    await cursor.execute(
                        "select t.* from dbms_vector_chain.utl_to_embeddings(:1, json(:2)) t",
                        [inputs, self._embedding_params],
                    )

                    for row in await cursor.fetchall():
                        if row is None:
                            embeddings.append([])
                        else:
                            rdata = json.loads(row[0])
                            # dereference string as array
                            vec = json.loads(rdata["embed_vector"])
                            embeddings.append(vec)
                except BaseException as embedding_error:
                    if proxy_was_set:
                        try:
                            await cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=None)
                        except Exception as cleanup_error:
                            msg = "Failed to clear Oracle session proxy after embedding failed"
                            logger.exception("%s; original embedding error was: %r", msg, embedding_error)
                            raise RuntimeError(msg) from cleanup_error
                    raise
                else:
                    if proxy_was_set:
                        try:
                            await cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=None)
                        except Exception as cleanup_error:
                            msg = "Failed to clear Oracle session proxy after embedding succeeded"
                            logger.exception(msg)
                            raise RuntimeError(msg) from cleanup_error

                return embeddings

        return await self._handle_context(context)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Compute an embedding for a single text string.

        :param text: The string to embed.
        :returns: A dictionary with:
            - embedding: The embedding of the input string.
            - meta: The embedding parameters used for the call (for example, provider, model, etc.).
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "OracleTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the OracleDocumentEmbedder."
            )
            raise TypeError(msg)

        return {"embedding": self._embed_documents([text])[0], "meta": self._embedding_params}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Asynchronously compute an embedding for a single text string.

        :param text: The string to embed.
        :returns: A dictionary with:
            - embedding: The embedding of the input string.
            - meta: The embedding parameters used for the call (for example, provider, model, etc.).
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "OracleTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the OracleDocumentEmbedder."
            )
            raise TypeError(msg)

        return {"embedding": (await self._embed_documents_async([text]))[0], "meta": self._embedding_params}

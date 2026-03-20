import array
import functools
import inspect
import re
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator, Literal, Optional, TypeVar, cast

import oracledb
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from .filters import _get_filter_string

logger = logging.getLogger(__name__)

DistanceStrategy = Literal["dot", "euclidean", "cosine"]
EmbeddingField = Literal["embedding", "sparse_embedding"]

VALID_DISTANCE_FUNCTIONS: tuple[DistanceStrategy, ...] = ("dot", "euclidean", "cosine")

# define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except oracledb.IntegrityError as integrity_error:
            (error_obj,) = integrity_error.args
            if error_obj.code == 1:
                logger.exception("Duplicate document error occurred.")
                raise DuplicateDocumentError() from integrity_error
            else:
                logger.exception("DB-related error occurred.")
                raise RuntimeError(f"Failed due to a DB error: {integrity_error}") from integrity_error
        except oracledb.Error as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(f"Failed due to a DB error: {db_err}") from db_err
        except RuntimeError as runtime_err:
            # Handle a runtime error
            logger.exception("Runtime error occurred.")
            raise RuntimeError(f"Failed due to a runtime error: {runtime_err}") from runtime_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError(f"Validation failed: {val_err}") from val_err
        except FilterError as filter_err:
            logger.exception("Filter error.")
            raise filter_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception(f"An unexpected error occurred: {e}")
            raise RuntimeError(f"Unexpected error: {e}") from e

    return cast(T, wrapper)


def _handle_exceptions_async(func: T) -> T:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except oracledb.IntegrityError as integrity_error:
            (error_obj,) = integrity_error.args
            if error_obj.code == 1:
                logger.exception("Duplicate document error occurred.")
                raise DuplicateDocumentError() from integrity_error
            else:
                logger.exception("DB-related error occurred.")
                raise RuntimeError(f"Failed due to a DB error: {integrity_error}") from integrity_error
        except oracledb.Error as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(f"Failed due to a DB error: {db_err}") from db_err
        except RuntimeError as runtime_err:
            # Handle a runtime error
            logger.exception("Runtime error occurred.")
            raise RuntimeError(f"Failed due to a runtime error: {runtime_err}") from runtime_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError(f"Validation failed: {val_err}") from val_err
        except FilterError as filter_err:
            logger.exception("Filter error.")
            raise filter_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception(f"An unexpected error occurred: {e}")
            raise RuntimeError(f"Unexpected error: {e}") from e

    return cast(T, wrapper)


@contextmanager
def _get_connection(client: Any) -> Iterator[oracledb.Connection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "ConnectionPool", None)

    if isinstance(client, oracledb.Connection):
        yield client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        with client.acquire() as connection:
            yield connection
    else:
        valid_types = "oracledb.Connection"
        if connection_pool_class:
            valid_types += " or oracledb.ConnectionPool"
        raise TypeError(f"Expected client of type {valid_types}, got {type(client).__name__}")


@asynccontextmanager
async def _get_connection_async(client: Any) -> AsyncIterator[oracledb.AsyncConnection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "AsyncConnectionPool", None)

    if isinstance(client, oracledb.AsyncConnection):
        yield client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        async with client.acquire() as connection:
            yield connection
    else:
        valid_types = "oracledb.AsyncConnection"
        if connection_pool_class:
            valid_types += " or oracledb.AsyncConnectionPool"
        raise TypeError(f"Expected client of type {valid_types}, got {type(client).__name__}")


def _compare_version(version: str, target_version: str) -> bool:
    # Split both version strings into parts
    version_parts = [int(part) for part in version.split(".")]
    target_parts = [int(part) for part in target_version.split(".")]

    # Compare each part
    for v, t in zip(version_parts, target_parts, strict=False):
        if v < t:
            return True  # Current version is less
        elif v > t:
            return False  # Current version is greater

    # If all parts equal so far, check if version has fewer parts than target_version
    return len(version_parts) < len(target_parts)


DUPLICATE_ERROR = 942


def _table_exists(connection: oracledb.Connection, table_name: str) -> bool:
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM {table_name} WHERE ROWNUM < 1")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == DUPLICATE_ERROR:
            return False
        raise


async def _table_exists_async(connection: oracledb.AsyncConnection, table_name: str) -> bool:
    try:
        with connection.cursor() as cursor:
            await cursor.execute(f"SELECT 1 FROM {table_name} WHERE ROWNUM < 1")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == DUPLICATE_ERROR:
            return False
        raise


def _get_table_dict(embedding_dim: int | None) -> dict[str, str]:
    cols_dict = {
        "id": "VARCHAR(128) PRIMARY KEY",
        "content": "CLOB",
        "blob_data": "BLOB",
        "blob_meta": "JSON",
        "blob_mime_type": "CLOB",
        "meta": "JSON",
        "score": "FLOAT",
        "embedding": f"vector({embedding_dim if embedding_dim else '*'}, FLOAT32)",
        "sparse_embedding": f"vector({embedding_dim if embedding_dim else '*'}, FLOAT32, SPARSE)",
    }
    return cols_dict


def _create_table(connection: oracledb.Connection, table_name: str, embedding_dim: int | None) -> None:
    cols_dict = _get_table_dict(embedding_dim)

    if not _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(f"{col_name} {col_type}" for col_name, col_type in cols_dict.items())
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            cursor.execute(ddl)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")


async def _create_table_async(
    connection: oracledb.AsyncConnection, table_name: str, embedding_dim: int | None
) -> None:
    cols_dict = _get_table_dict(embedding_dim)

    if not await _table_exists_async(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(f"{col_name} {col_type}" for col_name, col_type in cols_dict.items())
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            await cursor.execute(ddl)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")


# TODO
def _get_delete_ddl(table_name: str, ids: Optional[list[str]]) -> tuple[str, dict[str, str]]:
    if ids is None:
        raise ValueError("No ids provided to delete.")

    # constructing the SQL statement with individual placeholders
    placeholders = ", ".join([":id" + str(i + 1) for i in range(len(ids))])

    ddl = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"

    # preparing bind variables
    bind_vars = {f"id{i}": idx for i, idx in enumerate(ids, start=1)}

    return ddl, bind_vars


def _quote_indentifier(name: str) -> str:
    name = name.strip()
    reg = r'^(?:"[^"]+"|[^".]+)(?:\.(?:"[^"]+"|[^".]+))*$'
    pattern_validate = re.compile(reg)

    if not pattern_validate.match(name):
        raise ValueError(f"Identifier name {name} is not valid.")

    pattern_match = r'"([^"]+)"|([^".]+)'
    groups = re.findall(pattern_match, name)
    groups = [m[0] or m[1] for m in groups]
    groups = [f'"{g}"' for g in groups]

    return ".".join(groups)


################### INDEX


def _get_index_exists_query(index_name: str, table_name: Optional[str]) -> tuple[str, dict[str, str]]:
    query = f"""
        SELECT index_name
        FROM all_indexes
        WHERE index_name = :idx_name
        {"AND table_name = :table_name" if table_name else ""}
        """

    # this is an internal method, index_name and table_name comes with double quotes
    index_name = index_name.replace('"', "")
    parameters = {"idx_name": index_name}

    if table_name:
        table_name = table_name.replace('"', "")
        parameters["table_name"] = table_name

    return query, parameters


def _index_exists(connection: oracledb.Connection, index_name: str, table_name: Optional[str] = None) -> bool:
    # check if the index exists
    query, params = _get_index_exists_query(index_name, table_name)

    with connection.cursor() as cursor:
        # execute the query
        cursor.execute(query, **params)
        result = cursor.fetchone()

    return result is not None


async def _index_exists_async(
    connection: oracledb.AsyncConnection, index_name: str, table_name: Optional[str] = None
) -> bool:
    # check if the index exists
    query, params = _get_index_exists_query(index_name, table_name)

    with connection.cursor() as cursor:
        await cursor.execute(query, **params)
        result = await cursor.fetchone()

    return result is not None


def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f'"{base_name}_{unique_id}"'


def _get_hnsw_index_ddl(
    table_name: str,
    distance_strategy: str,
    params: Optional[dict[str, Any]] = None,
    embedding_column: Optional[str] = "embedding",
) -> tuple[str, str]:
    defaults = {
        "idx_name": "HNSW",
        "idx_type": "HNSW",
        "neighbors": 32,
        "efConstruction": 200,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(str(defaults[compulsory_key]))
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

    # base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"create vector index {idx_name} on {table_name}({embedding_column}) ORGANIZATION INMEMORY NEIGHBOR GRAPH"
    )

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {distance_strategy}"

    parameters_part = ""
    if "neighbors" in config and "efConstruction" in config:
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efConstruction {efConstruction})"
    elif "neighbors" in config and "efConstruction" not in config:
        config["efConstruction"] = defaults["efConstruction"]
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efConstruction {efConstruction})"
    elif "neighbors" not in config and "efConstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efConstruction {efConstruction})"

    # always included part for parallel
    parallel_part = " parallel {parallel}"

    # combine all parts
    ddl_assembly = base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    # format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    return idx_name, ddl


def _create_hnsw_index(
    connection: oracledb.Connection,
    table_name: str,
    distance_strategy: str,
    params: Optional[dict[str, Any]] = None,
    embedding_column: Optional[str] = "embedding",
) -> None:
    idx_name, ddl = _get_hnsw_index_ddl(table_name, distance_strategy, params, embedding_column)

    # check if the index exists
    if not _index_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


def _get_ivf_index_ddl(
    table_name: str,
    distance_strategy: str,
    params: Optional[dict[str, Any]] = None,
    embedding_column: Optional[str] = "embedding",
) -> tuple[str, str]:
    # default configuration
    defaults = {
        "idx_name": "IVF",
        "idx_type": "IVF",
        "neighbor_part": 32,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(str(defaults[compulsory_key]))
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

    # base SQL statement
    idx_name = config["idx_name"]
    base_sql = f"CREATE VECTOR INDEX {idx_name} ON {table_name}({embedding_column}) ORGANIZATION NEIGHBOR PARTITIONS"

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {distance_strategy}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_part" in config:
        parameters_part = f" PARAMETERS (type {config['idx_type']}, neighbor partitions {config['neighbor_part']})"

    # always included part for parallel
    parallel_part = f" PARALLEL {config['parallel']}"

    # combine all parts
    ddl_assembly = base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    # format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    return idx_name, ddl


def _create_ivf_index(
    connection: oracledb.Connection,
    table_name: str,
    distance_strategy: str,
    params: Optional[dict[str, Any]] = None,
    embedding_column: Optional[str] = "embedding",
) -> None:
    idx_name, ddl = _get_ivf_index_ddl(table_name, distance_strategy, params, embedding_column)

    # check if the index exists
    if not _index_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
        logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


async def _create_hnsw_index_async(
    connection: oracledb.AsyncConnection,
    table_name: str,
    distance_strategy: str,
    params: Optional[dict[str, Any]] = None,
    embedding_column: Optional[str] = "embedding",
) -> None:
    idx_name, ddl = _get_hnsw_index_ddl(table_name, distance_strategy, params, embedding_column)

    # check if the index exists
    if not await _index_exists_async(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
            logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


async def _create_ivf_index_async(
    connection: oracledb.AsyncConnection,
    table_name: str,
    distance_strategy: str,
    params: Optional[dict[str, Any]] = None,
    embedding_column: Optional[str] = "embedding",
) -> None:
    idx_name, ddl = _get_ivf_index_ddl(table_name, distance_strategy, params, embedding_column)

    # check if the index exists
    if not await _index_exists_async(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
        logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


def output_type_string_handler(cursor: Any, metadata: Any) -> Any:
    if metadata.type_code is oracledb.DB_TYPE_CLOB:
        return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_NCLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_BLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_RAW, arraysize=cursor.arraysize)


MERGE_QUERY = """
    MERGE INTO {table_name} t
    USING (SELECT :1 AS id, :2 AS content, :3 AS blob_data, :4 AS blob_meta,
                :5 AS blob_mime_type, :6 AS meta, :7 AS score, :8 AS embedding,
                :9 AS sparse_embedding FROM dual) s
    ON (t.id = s.id)
    WHEN MATCHED THEN
        UPDATE SET
            t.content = s.content,
            t.blob_data = s.blob_data,
            t.blob_meta = s.blob_meta,
            t.blob_mime_type = s.blob_mime_type,
            t.meta = s.meta,
            t.score = s.score,
            t.embedding = s.embedding,
            t.sparse_embedding = s.sparse_embedding
    WHEN NOT MATCHED THEN
        INSERT (id, content, blob_data, blob_meta,
            blob_mime_type, meta, score, embedding, sparse_embedding)
        VALUES (s.id, s.content, s.blob_data, s.blob_meta,
            s.blob_mime_type, s.meta, s.score, s.embedding, s.sparse_embedding);"""

INSERT_QUERY = """INSERT {policy} INTO {table_name}
    (id, content, blob_data, blob_meta, blob_mime_type, meta, score, embedding, sparse_embedding)
    VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9)"""


class OracleDocumentStore:
    """
    A document store using Oracle as the backend.
    """

    def __init__(
        self,
        connection_params: dict[str, Any],
        table_name: str = "documents",
        *,
        use_connection_pool: bool = False,
        embedding_dim: Optional[int] = None,
        create_vector_index: bool = False,
        vector_index_params: dict[str, Any] | None = None,
        vector_index_embedding_field: EmbeddingField = "embedding",
        vector_index_distance_strategy: DistanceStrategy = "cosine",
    ):
        """
        Create a new OracleDocumentStore instance.

        :param connection_params: Connection parameters for python-oracledb. These are passed to
            `oracledb.connect()`, `oracledb.connect_async()`, `oracledb.create_pool()`, or
            `oracledb.create_pool_async()` depending on the selected mode.
        :param table_name: Oracle table name used to store Haystack documents.
        :param use_connection_pool: If `True`, create and use an Oracle connection pool.
        :param embedding_dim: Optional dense and sparse embedding dimension for Oracle VECTOR columns.
            If omitted, the VECTOR columns are created with flexible dimensions.
        :param create_vector_index: If `True`, create a vector index during initialization.
        :param vector_index_params: Optional Oracle vector index parameters. Supported index types are `HNSW` and `IVF`.
        :param vector_index_embedding_field: VECTOR column to index. Must be either `embedding`
            or `sparse_embedding`.
        :param vector_index_distance_strategy: Distance strategy to use for vector indexing and retrieval.
            Must be one of `dot`, `euclidean`, or `cosine`.
        """

        # Store the params for marshalling
        self._connection_params = connection_params
        self._use_connection_pool = use_connection_pool
        self._table_name = _quote_indentifier(table_name)
        self._embedding_dim = embedding_dim
        self._create_vector_index = create_vector_index
        self._vector_index_params = vector_index_params
        if vector_index_distance_strategy not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{vector_index_distance_strategy}' for vector similarity. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        if vector_index_embedding_field not in ["embedding", "sparse_embedding"]:
            error_message = (
                f"Invalid field: '{vector_index_embedding_field}' for vector index. "
                f"Valid options are: {['embedding', 'sparse_embedding']}."
            )
            raise ValueError(error_message)

        self._vector_index_embedding_field: EmbeddingField = vector_index_embedding_field
        self._vector_index_distance_strategy: DistanceStrategy = vector_index_distance_strategy
        self._distance_strategy: DistanceStrategy = vector_index_distance_strategy
        self._initialized = False
        self._initialized_async = False
        self._client: Any = None
        self._client_async: Any = None

    def _require_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise ValueError("embedding_dim must be set for sparse embedding operations.")
        return self._embedding_dim

    def _ensure_initialized(self):
        if self._initialized:
            return

        if _compare_version(oracledb.__version__, "2.2.0"):
            raise Exception(
                f"Oracle DB python client driver version {oracledb.__version__} not supported, \
                must be >=2.2.0 for vector support"
            )

        if self._use_connection_pool:
            self._client = oracledb.create_pool(**self._connection_params)
        else:
            self._client = oracledb.connect(**self._connection_params)

        with _get_connection(self._client) as connection:
            table_exists = _table_exists(connection, self._table_name)

            if not table_exists:
                _create_table(
                    connection,
                    self._table_name,
                    self._embedding_dim,
                )

            if self._create_vector_index:
                self._create_index(connection, self._vector_index_params)

        self._initialized = True

    async def _ensure_initialized_async(self):
        if self._initialized_async:
            return

        if _compare_version(oracledb.__version__, "2.2.0"):
            raise Exception(
                f"Oracle DB python client driver version {oracledb.__version__} not supported, \
                must be >=2.2.0 for vector support"
            )

        if self._use_connection_pool:
            pool = cast(Any, oracledb.create_pool_async(**self._connection_params))
            self._client_async = await pool if inspect.isawaitable(pool) else pool
        else:
            self._client_async = await oracledb.connect_async(**self._connection_params)

        async def context(connection: oracledb.AsyncConnection) -> None:
            table_exists = await _table_exists_async(connection, self._table_name)

            if not table_exists:
                await _create_table_async(
                    connection,
                    self._table_name,
                    self._embedding_dim,
                )

            if self._create_vector_index:
                await self._create_index_async(connection, self._vector_index_params)

        await self._handle_context(context)

        self._initialized_async = True

    @_handle_exceptions
    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: how many documents are present in the document store.
        """
        self._ensure_initialized()
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT count(*) from {self._table_name}")
                res = cursor.fetchone()[0]

        return res

    @_handle_exceptions_async
    async def count_documents_async(self) -> int:
        """
        Asynchronously returns how many documents are present in the document store.

        :returns: how many documents are present in the document store.
        """
        await self._ensure_initialized_async()

        async def context(connection: oracledb.AsyncConnection) -> int:
            with connection.cursor() as cursor:
                await cursor.execute(f"SELECT count(*) from {self._table_name}")
                res = await cursor.fetchone()

            return res[0]

        return await self._handle_context(context)

    @_handle_exceptions
    def filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        self._ensure_initialized()

        where_clause = ""
        params: dict[str, Any] = {}
        if filters:
            bind_variables: list[Any] = []
            where_clause = _get_filter_string(filters, "meta", bind_variables)

            for i, value in enumerate(bind_variables):
                params[f"value{i}"] = value

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                db_query = f"""
                SELECT *
                FROM {self._table_name}
                {f"WHERE {where_clause}" if filters else ""}
                """

                cursor.execute(db_query, params)
                columns = [col.name for col in cursor.description]
                results = self._get_result_to_documents(cursor.fetchall(), columns)

        return results

    @_handle_exceptions_async
    async def filter_documents_async(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        where_clause = ""
        params: dict[str, Any] = {}
        if filters:
            bind_variables: list[Any] = []
            where_clause = _get_filter_string(filters, "meta", bind_variables)
            for i, value in enumerate(bind_variables):
                params[f"value{i}"] = value

        async def context(connection: oracledb.AsyncConnection) -> list[Document]:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                db_query = f"""
                SELECT *
                FROM {self._table_name}
                {f"WHERE {where_clause}" if filters else ""}
                """

                await cursor.execute(db_query, params)
                columns = [col.name for col in cursor.description]
                results = self._get_result_to_documents(await cursor.fetchall(), columns)

            return results

        return await self._handle_context(context)

    @_handle_exceptions
    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents:
            A list of documents to write into the document store.
        :param policy:
            Not supported at the moment.

        :raises ValueError:
            When input is not valid.

        :returns:
            The number of documents written
        """
        if len(documents) > 0:
            if any(not isinstance(doc, Document) for doc in documents):
                msg = "'documents' must contain a list of Document"
                raise ValueError(msg)

        self._ensure_initialized()
        embedding_dim = self._require_embedding_dim()
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, None, None, oracledb.DB_TYPE_JSON, None, oracledb.DB_TYPE_JSON, None, None, None
                )

                if policy == DuplicatePolicy.OVERWRITE:
                    query = MERGE_QUERY.format(table_name=self._table_name)

                else:
                    if policy == DuplicatePolicy.SKIP:
                        policy_hint = f"/*+ ignore_row_on_dupkey_index({self._table_name}(id)) */"
                    else:
                        policy_hint = ""
                    query = INSERT_QUERY.format(table_name=self._table_name, policy=policy_hint)

                bind_input = [
                    (
                        doc.id,
                        doc.content or None,
                        doc.blob.data if doc.blob else None,
                        doc.blob.meta if doc.blob else None,
                        doc.blob.mime_type if doc.blob and doc.blob.mime_type else None,
                        doc.meta or None,
                        doc.score or None,
                        array.array("f", doc.embedding) if doc.embedding else None,
                        oracledb.SparseVector(
                            embedding_dim,
                            doc.sparse_embedding.indices,
                            array.array("f", doc.sparse_embedding.values),
                        )
                        if doc.sparse_embedding
                        else None,
                    )
                    for doc in documents
                ]

                cursor.executemany(
                    query,
                    bind_input,
                )
                connection.commit()
                num_docs = cursor.rowcount

        return num_docs

    @_handle_exceptions_async
    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
        """
        Asynchronously writes (or overwrites) documents into the store.

        :param documents:
            A list of documents to write into the document store.
        :param policy:
            Not supported at the moment.

        :raises ValueError:
            When input is not valid.

        :returns:
            The number of documents written
        """
        if len(documents) > 0:
            if any(not isinstance(doc, Document) for doc in documents):
                msg = "'documents' must contain a list of Document"
                raise ValueError(msg)

        await self._ensure_initialized_async()
        embedding_dim = self._require_embedding_dim()

        async def context(
            connection: oracledb.AsyncConnection,
        ) -> int:
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, None, None, oracledb.DB_TYPE_JSON, None, oracledb.DB_TYPE_JSON, None, None, None
                )

                if policy == DuplicatePolicy.OVERWRITE:
                    query = MERGE_QUERY.format(table_name=self._table_name)

                else:
                    if policy == DuplicatePolicy.SKIP:
                        policy_hint = f"/*+ ignore_row_on_dupkey_index({self._table_name}(id)) */"
                    else:
                        policy_hint = ""
                    query = INSERT_QUERY.format(table_name=self._table_name, policy=policy_hint)

                bind_input = [
                    (
                        doc.id,
                        doc.content or None,
                        doc.blob.data if doc.blob else None,
                        doc.blob.meta if doc.blob else None,
                        doc.blob.mime_type if doc.blob and doc.blob.mime_type else None,
                        doc.meta or None,
                        doc.score or None,
                        array.array("f", doc.embedding) if doc.embedding else None,
                        oracledb.SparseVector(
                            embedding_dim,
                            doc.sparse_embedding.indices,
                            array.array("f", doc.sparse_embedding.values),
                        )
                        if doc.sparse_embedding
                        else None,
                    )
                    for doc in documents
                ]

                await cursor.executemany(
                    query,
                    bind_input,
                )
                await connection.commit()

                return cursor.rowcount

        num_docs = await self._handle_context(context)
        return num_docs

    @_handle_exceptions
    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        self._ensure_initialized()
        ddl, bind_vars = _get_delete_ddl(self._table_name, document_ids)

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl, bind_vars)
                connection.commit()

    @_handle_exceptions_async
    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        await self._ensure_initialized_async()
        ddl, bind_vars = _get_delete_ddl(self._table_name, document_ids)

        async def context(
            connection: oracledb.AsyncConnection,
        ) -> None:
            with connection.cursor() as cursor:
                await cursor.execute(ddl, bind_vars)
                await connection.commit()

        await self._handle_context(context)

    @staticmethod
    def _get_result_to_documents(result: list, columns: list[str]) -> list[Document]:
        """
        Helper function to convert Oracle query results into Haystack Documents.
        """
        retval = []
        for res in result:
            document_dict: dict[str, Any] = {}
            blob_dict = {}
            for i, col in enumerate(columns):
                col_l = col.lower()

                if col_l == "sparse_embedding" and res[i]:
                    document_dict[col_l] = {"indices": res[i].indices, "values": res[i].values}
                elif col_l == "blob_data":
                    blob_dict["data"] = res[i]
                elif col_l == "blob_meta":
                    blob_dict["meta"] = res[i]
                elif col_l == "blob_mime_type":
                    blob_dict["mime_type"] = res[i]
                else:
                    document_dict[col_l] = res[i]

            if blob_dict["data"]:
                document_dict["blob"] = blob_dict
            else:
                document_dict["blob"] = None

            if not document_dict["meta"]:
                document_dict["meta"] = {}

            retval.append(Document.from_dict(document_dict))

        return retval

    async def _handle_context(
        self,
        context: Callable[
            [oracledb.AsyncConnection],
            Any,
        ],
    ) -> Any:
        """Acquire an async Oracle connection and release it when the callback completes."""
        async with _get_connection_async(self._client_async) as connection:
            return await context(connection)

    def _embedding_retrieval(
        self,
        query_embedding: list[float] | SparseEmbedding,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = "cosine",
    ) -> list[Document]:
        self._ensure_initialized()
        where_clause = ""
        params: dict[str, Any]

        if isinstance(query_embedding, SparseEmbedding):
            embedding_dim = self._require_embedding_dim()
            column = "sparse_embedding"
            params = {
                "embedding": oracledb.SparseVector(
                    embedding_dim,
                    query_embedding.indices,
                    array.array("f", query_embedding.values),
                )
            }
        else:
            column = "embedding"
            params = {"embedding": array.array("f", query_embedding)}

        if filters:
            bind_variables: list[Any] = []
            where_clause = _get_filter_string(filters, "meta", bind_variables)
            for i, value in enumerate(bind_variables):
                params[f"value{i}"] = value

        db_query = f"""
            SELECT *
            FROM {self._table_name}
            {f"WHERE {where_clause}" if filters else ""}
            ORDER BY
            vector_distance({column}, :embedding, {distance_strategy})
            FETCH APPROX FIRST {top_k} ROWS ONLY
            """

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                cursor.execute(db_query, params)
                columns = [col.name for col in cursor.description]
                results = self._get_result_to_documents(cursor.fetchall(), columns)

        return results

    async def _embedding_retrieval_async(
        self,
        query_embedding: list[float] | SparseEmbedding,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = "cosine",
    ) -> list[Document]:
        await self._ensure_initialized_async()

        async def context(
            connection: oracledb.AsyncConnection,
        ) -> list[Document]:
            where_clause = ""
            params: dict[str, Any]

            if isinstance(query_embedding, SparseEmbedding):
                embedding_dim = self._require_embedding_dim()
                column = "sparse_embedding"
                params = {
                    "embedding": oracledb.SparseVector(
                        embedding_dim,
                        query_embedding.indices,
                        array.array("f", query_embedding.values),
                    )
                }
            else:
                column = "embedding"
                params = {"embedding": array.array("f", query_embedding)}

            if filters:
                bind_variables: list[Any] = []
                where_clause = _get_filter_string(filters, "meta", bind_variables)
                for i, value in enumerate(bind_variables):
                    params[f"value{i}"] = value

            db_query = f"""
                SELECT *
                FROM {self._table_name}
                {f"WHERE {where_clause}" if filters else ""}
                ORDER BY
                vector_distance({column}, :embedding, {distance_strategy})
                FETCH APPROX FIRST {top_k} ROWS ONLY
                """

            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                await cursor.execute(db_query, params)
                columns = [col.name for col in cursor.description]
                results = self._get_result_to_documents(await cursor.fetchall(), columns)

            return results

        return await self._handle_context(context)

    async def _create_index_async(
        self, connection: oracledb.AsyncConnection, params: dict[str, Any] | None = None
    ) -> None:
        if params and "idx_name" in params:
            params["idx_name"] = _quote_indentifier(params["idx_name"])

        if params:
            if params["idx_type"] == "HNSW":
                await _create_hnsw_index_async(
                    connection, self._table_name, self._distance_strategy, params, self._vector_index_embedding_field
                )
            elif params["idx_type"] == "IVF":
                await _create_ivf_index_async(
                    connection, self._table_name, self._distance_strategy, params, self._vector_index_embedding_field
                )
            else:
                raise ValueError("Only supported indexes HNSW and IVF")
        else:
            await _create_hnsw_index_async(
                connection, self._table_name, self._distance_strategy, params, self._vector_index_embedding_field
            )

    def _create_index(self, connection: oracledb.Connection, params: dict[str, Any] | None = None) -> None:
        if params and "idx_name" in params:
            params["idx_name"] = _quote_indentifier(params["idx_name"])

        if params:
            if params["idx_type"] == "HNSW":
                _create_hnsw_index(
                    connection, self._table_name, self._distance_strategy, params, self._vector_index_embedding_field
                )
            elif params["idx_type"] == "IVF":
                _create_ivf_index(
                    connection, self._table_name, self._distance_strategy, params, self._vector_index_embedding_field
                )
            else:
                raise ValueError("Only supported indexes HNSW and IVF")
        else:
            _create_hnsw_index(
                connection, self._table_name, self._distance_strategy, params, self._vector_index_embedding_field
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_params=self._connection_params,
            table_name=self._table_name,
            use_connection_pool=self._use_connection_pool,
            embedding_dim=self._embedding_dim,
            create_vector_index=self._create_vector_index,
            vector_index_params=self._vector_index_params,
            vector_index_embedding_field=self._vector_index_embedding_field,
            vector_index_distance_strategy=self._vector_index_distance_strategy,
        )

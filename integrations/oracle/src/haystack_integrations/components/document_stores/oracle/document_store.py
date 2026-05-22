import array
import functools
import importlib
import inspect
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, cast

import oracledb
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.utils import Secret, deserialize_secrets_inplace

from .filters import _get_filter_string

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haystack_integrations.components.embedders.oracle import OracleTextEmbedder

DistanceStrategy = Literal["dot", "euclidean", "cosine"]
EmbeddingField = Literal["embedding", "sparse_embedding"]

VALID_DISTANCE_FUNCTIONS: tuple[DistanceStrategy, ...] = ("dot", "euclidean", "cosine")
_PASSWORD_IN_CONNECT_STRING = re.compile(r"^\S+/\S+@\S+$")
_PASSWORD_IN_URL = re.compile(r"://[^/@\s:]+:[^/@\s]+@")
_SENSITIVE_CONNECTION_KEY_PARTS = (
    "user",
    "username",
    "password",
    "passwd",
    "pwd",
    "dsn",
    "secret",
    "token",
    "key",
    "credential",
)

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

BASE_DOCUMENT_COLUMNS = [
    "id",
    "content",
    "blob_data",
    "blob_meta",
    "blob_mime_type",
    "meta",
    "score",
    "embedding",
]


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


def _get_table_dict(embedding_dim: int | None, *, support_sparse_embeddings: bool) -> dict[str, str]:
    cols_dict = {
        "id": "VARCHAR(128) PRIMARY KEY",
        "content": "CLOB",
        "blob_data": "BLOB",
        "blob_meta": "JSON",
        "blob_mime_type": "CLOB",
        "meta": "JSON",
        "score": "FLOAT",
        "embedding": f"vector({embedding_dim if embedding_dim else '*'}, FLOAT32)",
    }
    if support_sparse_embeddings:
        cols_dict["sparse_embedding"] = f"vector({embedding_dim if embedding_dim else '*'}, FLOAT32, SPARSE)"
    return cols_dict


def _create_table(
    connection: oracledb.Connection,
    table_name: str,
    embedding_dim: int | None,
    *,
    support_sparse_embeddings: bool,
) -> None:
    cols_dict = _get_table_dict(embedding_dim, support_sparse_embeddings=support_sparse_embeddings)

    if not _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(f"{col_name} {col_type}" for col_name, col_type in cols_dict.items())
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            cursor.execute(ddl)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")


async def _create_table_async(
    connection: oracledb.AsyncConnection,
    table_name: str,
    embedding_dim: int | None,
    *,
    support_sparse_embeddings: bool,
) -> None:
    cols_dict = _get_table_dict(embedding_dim, support_sparse_embeddings=support_sparse_embeddings)

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


def _quote_identifier(name: str) -> str:
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


def _validate_int_param(
    config: dict[str, Any],
    key: str,
    min_value: int,
    max_value: int | None = None,
) -> None:
    if key not in config:
        return

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value < min_value:
        raise ValueError(f"{key} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{key} must be at most {max_value}.")


def _validate_allowed_params(config: dict[str, Any], allowed_keys: set[str]) -> None:
    for key in config:
        if key not in allowed_keys:
            raise ValueError(f"Invalid parameter: {key}")


def _validate_index_type(config: dict[str, Any], expected_type: str) -> None:
    if "idx_type" not in config:
        return

    idx_type = config["idx_type"]
    if not isinstance(idx_type, str) or idx_type.upper() != expected_type:
        raise ValueError(f"idx_type must be {expected_type}.")
    config["idx_type"] = expected_type


def _quote_hybrid_identifier(value: str, field_name: str) -> str:
    value = value.strip()
    simple_identifier = r"[A-Za-z][A-Za-z0-9_$#]*"
    reg = rf'^(?:"{simple_identifier}"|{simple_identifier})(?:\.(?:"{simple_identifier}"|{simple_identifier}))*$'
    if not re.fullmatch(reg, value):
        raise ValueError(f"{field_name} contains an invalid identifier.")

    pattern_match = rf'"({simple_identifier})"|({simple_identifier})'
    groups = re.findall(pattern_match, value)
    quoted_groups = [f'"{quoted}"' if quoted else f'"{unquoted.upper()}"' for quoted, unquoted in groups]
    return ".".join(quoted_groups)


def _quote_hybrid_identifier_list(values: Any, field_name: str) -> str:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of column names.")
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} must contain only column names.")
    return ",".join(_quote_hybrid_identifier(value, field_name) for value in values)


def _is_sensitive_connection_key(key: str) -> bool:
    key_lower = key.lower()
    return any(part in key_lower for part in _SENSITIVE_CONNECTION_KEY_PARTS)


def _is_sensitive_connection_string(value: str) -> bool:
    return bool(_PASSWORD_IN_CONNECT_STRING.fullmatch(value) or _PASSWORD_IN_URL.search(value))


def _serialize_connection_param(key: str, value: Any) -> Any:
    if isinstance(value, Secret):
        return value.to_dict()
    if _is_sensitive_connection_key(key):
        return None
    if isinstance(value, str) and _is_sensitive_connection_string(value):
        return None
    return value


def _serialize_connection_params(connection_params: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_connection_param(key, value) for key, value in connection_params.items()}


def _deserialize_connection_params(connection_params: dict[str, Any]) -> None:
    secret_keys = [
        key
        for key, value in connection_params.items()
        if isinstance(value, dict) and value.get("type") in {"env_var", "token"}
    ]
    if secret_keys:
        deserialize_secrets_inplace(connection_params, keys=secret_keys)


def _resolve_connection_params(connection_params: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in connection_params.items():
        if isinstance(value, Secret):
            resolved[key] = value.resolve_value()
        else:
            resolved[key] = value
    return resolved


def _serialize_optional_secret(value: Secret | str | None) -> dict[str, Any] | None:
    if isinstance(value, Secret):
        return value.to_dict()
    return None


def _deserialize_optional_secret(data: dict[str, Any], key: str) -> None:
    if isinstance(data.get(key), dict) and data[key].get("type") in {"env_var", "token"}:
        deserialize_secrets_inplace(data, keys=[key])


def _resolve_optional_secret(value: Secret | str | None) -> str | None:
    if isinstance(value, Secret):
        return value.resolve_value()
    return value


def _validate_text_index_column(column_name: str) -> str:
    normalized = column_name.strip().lower()
    if normalized != "content":
        raise ValueError("Oracle text indexing currently supports only the 'content' column.")
    return normalized


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


def _get_text_index_ddl(table_name: str, idx_name: str, column_name: str = "content") -> str:
    validated_column = _validate_text_index_column(column_name)
    return f"CREATE SEARCH INDEX {idx_name} ON {table_name}({validated_column})"


def _create_text_index(
    connection: oracledb.Connection,
    table_name: str,
    idx_name: str,
    column_name: str = "content",
) -> None:
    ddl = _get_text_index_ddl(table_name, idx_name, column_name)
    if not _index_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            logger.info(f"Text index {idx_name} created successfully...")
    else:
        logger.info(f"Text index {idx_name} already exists...")


async def _create_text_index_async(
    connection: oracledb.AsyncConnection,
    table_name: str,
    idx_name: str,
    column_name: str = "content",
) -> None:
    ddl = _get_text_index_ddl(table_name, idx_name, column_name)
    if not await _index_exists_async(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
            logger.info(f"Text index {idx_name} created successfully...")
    else:
        logger.info(f"Text index {idx_name} already exists...")


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
        "efconstruction": 200,
        "accuracy": 90,
        "parallel": 8,
    }

    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary.")
        config = params.copy()
        # ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(str(defaults[compulsory_key]))
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        _validate_allowed_params(config, set(defaults))
    else:
        config = defaults.copy()
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

    if ("neighbors" in config or "efconstruction" in config) and "idx_type" not in config:
        config["idx_type"] = defaults["idx_type"]
    _validate_index_type(config, "HNSW")
    _validate_int_param(config, "accuracy", 1, 100)
    _validate_int_param(config, "neighbors", 2, 2048)
    _validate_int_param(config, "efconstruction", 1, 65535)
    _validate_int_param(config, "parallel", 1)

    # base SQL statement
    config["idx_name"] = _quote_identifier(config["idx_name"])
    idx_name = config["idx_name"]
    base_sql = (
        f"create vector index {idx_name} on {table_name}({embedding_column}) ORGANIZATION INMEMORY NEIGHBOR GRAPH"
    )

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {distance_strategy}"

    parameters_part = ""
    if "neighbors" in config and "efconstruction" in config:
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efconstruction {efconstruction})"
    elif "neighbors" in config and "efconstruction" not in config:
        config["efconstruction"] = defaults["efconstruction"]
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efconstruction {efconstruction})"
    elif "neighbors" not in config and "efconstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efconstruction {efconstruction})"

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
        "neighbor_partitions": 32,
        "accuracy": 90,
        "parallel": 8,
    }
    allowed_keys = set(defaults) | {"samples_per_partition", "min_vectors_per_partition"}

    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary.")
        config = params.copy()
        # ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(str(defaults[compulsory_key]))
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        _validate_allowed_params(config, allowed_keys)
    else:
        config = defaults.copy()
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

    if {
        "neighbor_partitions",
        "samples_per_partition",
        "min_vectors_per_partition",
    } & set(config) and "idx_type" not in config:
        config["idx_type"] = defaults["idx_type"]
    _validate_index_type(config, "IVF")
    _validate_int_param(config, "accuracy", 1, 100)
    _validate_int_param(config, "neighbor_partitions", 1, 10000000)
    _validate_int_param(config, "samples_per_partition", 1)
    _validate_int_param(config, "min_vectors_per_partition", 0)
    _validate_int_param(config, "parallel", 1)

    # base SQL statement
    config["idx_name"] = _quote_identifier(config["idx_name"])
    idx_name = config["idx_name"]
    base_sql = f"CREATE VECTOR INDEX {idx_name} ON {table_name}({embedding_column}) ORGANIZATION NEIGHBOR PARTITIONS"

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {distance_strategy}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_partitions" in config:
        parameters_part = f" PARAMETERS (type {config['idx_type']}, neighbor partitions {config['neighbor_partitions']}"
        if "samples_per_partition" in config:
            parameters_part += f", samples_per_partition {config['samples_per_partition']}"
        if "min_vectors_per_partition" in config:
            parameters_part += f", min_vectors_per_partition {config['min_vectors_per_partition']}"
        parameters_part += ")"

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


def _get_document_columns(*, support_sparse_embeddings: bool) -> list[str]:
    columns = list(BASE_DOCUMENT_COLUMNS)
    if support_sparse_embeddings:
        columns.append("sparse_embedding")
    return columns


def _get_insert_query(table_name: str, policy: str, *, support_sparse_embeddings: bool) -> str:
    columns = _get_document_columns(support_sparse_embeddings=support_sparse_embeddings)
    placeholders = ", ".join(f":{i}" for i in range(1, len(columns) + 1))
    return f"""INSERT {policy} INTO {table_name}
    ({", ".join(columns)})
    VALUES ({placeholders})"""


def _get_merge_query(table_name: str, *, support_sparse_embeddings: bool) -> str:
    columns = _get_document_columns(support_sparse_embeddings=support_sparse_embeddings)
    source_select = ", ".join(f":{i} AS {column}" for i, column in enumerate(columns, start=1))
    update_columns = ",\n            ".join(f"t.{column} = s.{column}" for column in columns if column != "id")
    insert_columns = ", ".join(columns)
    insert_values = ", ".join(f"s.{column}" for column in columns)
    return f"""
    MERGE INTO {table_name} t
    USING (SELECT {source_select} FROM dual) s
    ON (t.id = s.id)
    WHEN MATCHED THEN
        UPDATE SET
            {update_columns}
    WHEN NOT MATCHED THEN
        INSERT ({insert_columns})
        VALUES ({insert_values});"""


def _normalize_sparse_vector_index_config(
    sparse_vector_index: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if sparse_vector_index is None:
        return None

    valid_keys = {"enabled", "distance_strategy", "params"}
    invalid_keys = set(sparse_vector_index) - valid_keys
    if invalid_keys:
        invalid_keys_text = ", ".join(sorted(invalid_keys))
        raise ValueError(f"Invalid sparse_vector_index parameter(s): {invalid_keys_text}")

    enabled = sparse_vector_index.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("sparse_vector_index.enabled must be a boolean.")

    normalized: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        return normalized

    distance_strategy = sparse_vector_index.get("distance_strategy", "cosine")
    if distance_strategy not in VALID_DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Invalid distance_function: '{distance_strategy}' for vector similarity. "
            f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
        )

    params = sparse_vector_index.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError("sparse_vector_index.params must be a dictionary when provided.")

    normalized["distance_strategy"] = distance_strategy
    normalized["params"] = params
    return normalized


def _validate_vectorizer_parameters(
    embedding_params: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    if "model" in params:
        model_name = params["model"]
        if embedding_params.get("provider") != "database" or embedding_params.get("model") != model_name:
            raise ValueError(
                "Mismatch between text_embedder and provided params: expected "
                f"provider='database' and model='{embedding_params.get('model')}', "
                f"but received model='{model_name}'."
            )
        return True

    if "embedder_spec" in params:
        if json.dumps(embedding_params, sort_keys=True) != json.dumps(params["embedder_spec"], sort_keys=True):
            raise ValueError(
                "Mismatch between text_embedder and provided params: embedder_spec must exactly match "
                "text_embedder.embedding_params after JSON normalization."
            )
        return True

    return False


def _get_vectorizer_preference_parameters(
    text_embedder: "OracleTextEmbedder",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preference_params = params.copy() if params else {}
    embedding_params = text_embedder._embedding_params
    has_model_config = _validate_vectorizer_parameters(embedding_params, preference_params)

    if not has_model_config:
        if embedding_params.get("provider") == "database":
            preference_params["model"] = embedding_params.get("model")
        else:
            preference_params["embedder_spec"] = embedding_params

    return preference_params


def _validate_text_embedder_instance(text_embedder: Any) -> None:
    oracle_embedder_module = importlib.import_module("haystack_integrations.components.embedders.oracle")
    oracle_text_embedder = oracle_embedder_module.OracleTextEmbedder
    if not isinstance(text_embedder, oracle_text_embedder):
        raise ValueError("text_embedder must be an instance of OracleTextEmbedder")


class OracleVectorizerPreference:
    """Manage DBMS_VECTOR_CHAIN vectorizer preferences for Oracle hybrid indexes."""

    PREFERENCE_CREATE_DDL = """
    begin
    dbms_vector_chain.CREATE_PREFERENCE(
        :1,
        dbms_vector_chain.vectorizer,
        json(:2));
    end;"""

    PREFERENCE_DROP_DDL = "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;"

    def __init__(self, document_store: "OracleDocumentStore", preference_name: str):
        self.document_store = document_store
        self.preference_name = preference_name

    @classmethod
    @_handle_exceptions
    def create(
        cls,
        document_store: "OracleDocumentStore",
        text_embedder: "OracleTextEmbedder",
        preference_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> "OracleVectorizerPreference":
        if not isinstance(document_store, OracleDocumentStore):
            raise ValueError("document_store must be an instance of OracleDocumentStore")
        _validate_text_embedder_instance(text_embedder)

        preference = cls(document_store, preference_name or f"pref{uuid.uuid4().hex[:15]}")
        preference_params = _get_vectorizer_preference_parameters(text_embedder, params)

        document_store._ensure_initialized()
        with _get_connection(document_store._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    cls.PREFERENCE_CREATE_DDL,
                    [preference.preference_name, json.dumps(preference_params)],
                )
        return preference

    @classmethod
    @_handle_exceptions_async
    async def create_async(
        cls,
        document_store: "OracleDocumentStore",
        text_embedder: "OracleTextEmbedder",
        preference_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> "OracleVectorizerPreference":
        if not isinstance(document_store, OracleDocumentStore):
            raise ValueError("document_store must be an instance of OracleDocumentStore")
        _validate_text_embedder_instance(text_embedder)

        preference = cls(document_store, preference_name or f"pref{uuid.uuid4().hex[:15]}")
        preference_params = _get_vectorizer_preference_parameters(text_embedder, params)

        await document_store._ensure_initialized_async()
        async with _get_connection_async(document_store._client_async) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(
                    cls.PREFERENCE_CREATE_DDL,
                    [preference.preference_name, json.dumps(preference_params)],
                )
        return preference

    @_handle_exceptions
    def drop(self) -> None:
        self.document_store._ensure_initialized()
        with _get_connection(self.document_store._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(self.PREFERENCE_DROP_DDL, preference_name=self.preference_name)

    @_handle_exceptions_async
    async def drop_async(self) -> None:
        await self.document_store._ensure_initialized_async()
        async with _get_connection_async(self.document_store._client_async) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(self.PREFERENCE_DROP_DDL, preference_name=self.preference_name)


def _get_hybrid_index_ddl(
    table_name: str,
    idx_name: str,
    vectorizer_preference: OracleVectorizerPreference,
    params: dict[str, Any] | None = None,
) -> str:
    params = params or {}
    index_parameters = params.get("parameters", {}).copy()
    if any(key.lower() in {"model", "embedder_spec", "vectorizer", "vector_idxtype"} for key in index_parameters):
        raise ValueError(
            "Vectorization parameters must be given with OracleVectorizerPreference: do not include any of "
            "{model, embedder_spec, vectorizer, vector_idxtype} under params['parameters']."
        )

    params_str = f"vectorizer {vectorizer_preference.preference_name} "
    for key, value in index_parameters.items():
        params_str += f"{key} {value} "

    filter_by_str = ""
    filter_by = params.get("filter_by")
    if filter_by:
        filter_by_str = "FILTER BY " + _quote_hybrid_identifier_list(filter_by, "filter_by") + " "

    order_by_str = ""
    order_by = params.get("order_by")
    order_by_asc = params.get("order_by_asc", True)
    if not isinstance(order_by_asc, bool):
        raise ValueError("order_by_asc must be a boolean.")
    if order_by:
        order_by_str = (
            "ORDER BY " + _quote_hybrid_identifier_list(order_by, "order_by") + f" {'ASC' if order_by_asc else 'DESC'} "
        )

    parallel_str = ""
    parallel = params.get("parallel")
    if parallel is not None:
        if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel <= 0:
            raise ValueError("parallel must be a positive integer.")
        parallel_str = f"PARALLEL {parallel} "

    escaped_params_str = params_str.replace("'", "''")
    return (
        f"CREATE HYBRID VECTOR INDEX {idx_name} ON {table_name}(content) "
        f"PARAMETERS ('{escaped_params_str}') "
        f"{filter_by_str}{order_by_str}{parallel_str}"
    )


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
        support_sparse_embeddings: bool = True,
        create_vector_index: bool = False,
        vector_index_params: dict[str, Any] | None = None,
        vector_index_embedding_field: EmbeddingField = "embedding",
        vector_index_distance_strategy: DistanceStrategy = "cosine",
        sparse_vector_index: dict[str, Any] | None = None,
    ):
        """
        Create a new OracleDocumentStore instance.

        :param connection_params: Connection parameters for python-oracledb. These are passed to
            `oracledb.connect()`, `oracledb.connect_async()`, `oracledb.create_pool()`, or
            `oracledb.create_pool_async()` depending on the selected mode. Values can be Haystack `Secret`
            instances to avoid serializing raw credentials.
        :param table_name: Oracle table name used to store Haystack documents.
        :param use_connection_pool: If `True`, create and use an Oracle connection pool.
        :param embedding_dim: Optional dense and sparse embedding dimension for Oracle VECTOR columns.
            If omitted, the VECTOR columns are created with flexible dimensions.
        :param support_sparse_embeddings: If `True`, create support for sparse embeddings in the table schema
            and allow sparse retrieval and writes.
        :param create_vector_index: If `True`, create a vector index during initialization.
        :param vector_index_params: Optional Oracle vector index parameters. Supported index types are `HNSW` and `IVF`.
        :param vector_index_embedding_field: VECTOR column to index. Must be either `embedding`
            or `sparse_embedding`.
        :param vector_index_distance_strategy: Distance strategy to use for vector indexing and retrieval.
            Must be one of `dot`, `euclidean`, or `cosine`.
        :param sparse_vector_index: Optional sparse vector index configuration. Supported keys are
            `enabled`, `distance_strategy`, and `params`.
        """

        # Store the params for marshalling
        self._connection_params = connection_params
        self._use_connection_pool = use_connection_pool
        self._table_name = _quote_identifier(table_name)
        self._embedding_dim = embedding_dim
        self._support_sparse_embeddings = support_sparse_embeddings
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

        if not support_sparse_embeddings and vector_index_embedding_field == "sparse_embedding":
            raise ValueError("vector_index_embedding_field='sparse_embedding' requires support_sparse_embeddings=True.")

        self._sparse_vector_index = _normalize_sparse_vector_index_config(sparse_vector_index)
        if not support_sparse_embeddings and self._sparse_vector_index and self._sparse_vector_index.get("enabled"):
            raise ValueError("sparse_vector_index.enabled requires support_sparse_embeddings=True.")

        if (
            self._sparse_vector_index
            and self._sparse_vector_index.get("enabled")
            and create_vector_index
            and vector_index_embedding_field == "sparse_embedding"
        ):
            raise ValueError(
                "Configure sparse index either with sparse_vector_index or with the legacy "
                "vector_index_embedding_field='sparse_embedding', not both."
            )

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

        resolved_connection_params = _resolve_connection_params(self._connection_params)
        if self._use_connection_pool:
            self._client = oracledb.create_pool(**resolved_connection_params)
        else:
            self._client = oracledb.connect(**resolved_connection_params)

        with _get_connection(self._client) as connection:
            table_exists = _table_exists(connection, self._table_name)

            if not table_exists:
                _create_table(
                    connection,
                    self._table_name,
                    self._embedding_dim,
                    support_sparse_embeddings=self._support_sparse_embeddings,
                )

            if self._create_vector_index:
                self._create_index(connection, self._vector_index_params)

            if self._sparse_vector_index and self._sparse_vector_index.get("enabled"):
                self._create_sparse_index(
                    connection,
                    self._sparse_vector_index.get("params"),
                    self._sparse_vector_index["distance_strategy"],
                )

        self._initialized = True

    async def _ensure_initialized_async(self):
        if self._initialized_async:
            return

        if _compare_version(oracledb.__version__, "2.2.0"):
            raise Exception(
                f"Oracle DB python client driver version {oracledb.__version__} not supported, \
                must be >=2.2.0 for vector support"
            )

        resolved_connection_params = _resolve_connection_params(self._connection_params)
        if self._use_connection_pool:
            pool = cast(Any, oracledb.create_pool_async(**resolved_connection_params))
            self._client_async = await pool if inspect.isawaitable(pool) else pool
        else:
            self._client_async = await oracledb.connect_async(**resolved_connection_params)

        async def context(connection: oracledb.AsyncConnection) -> None:
            table_exists = await _table_exists_async(connection, self._table_name)

            if not table_exists:
                await _create_table_async(
                    connection,
                    self._table_name,
                    self._embedding_dim,
                    support_sparse_embeddings=self._support_sparse_embeddings,
                )

            if self._create_vector_index:
                await self._create_index_async(connection, self._vector_index_params)

            if self._sparse_vector_index and self._sparse_vector_index.get("enabled"):
                await self._create_sparse_index_async(
                    connection,
                    self._sparse_vector_index.get("params"),
                    self._sparse_vector_index["distance_strategy"],
                )

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

        if not self._support_sparse_embeddings and any(doc.sparse_embedding for doc in documents):
            raise ValueError("Sparse embeddings are not supported by this document store.")

        self._ensure_initialized()
        embedding_dim = self._require_embedding_dim() if any(doc.sparse_embedding for doc in documents) else None
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                input_sizes = [None, None, None, oracledb.DB_TYPE_JSON, None, oracledb.DB_TYPE_JSON, None, None]
                if self._support_sparse_embeddings:
                    input_sizes.append(None)
                cursor.setinputsizes(*input_sizes)

                if policy == DuplicatePolicy.OVERWRITE:
                    query = _get_merge_query(
                        self._table_name, support_sparse_embeddings=self._support_sparse_embeddings
                    )

                else:
                    if policy == DuplicatePolicy.SKIP:
                        policy_hint = f"/*+ ignore_row_on_dupkey_index({self._table_name}(id)) */"
                    else:
                        policy_hint = ""
                    query = _get_insert_query(
                        self._table_name,
                        policy_hint,
                        support_sparse_embeddings=self._support_sparse_embeddings,
                    )

                bind_input = []
                for doc in documents:
                    row = [
                        doc.id,
                        doc.content or None,
                        doc.blob.data if doc.blob else None,
                        doc.blob.meta if doc.blob else None,
                        doc.blob.mime_type if doc.blob and doc.blob.mime_type else None,
                        doc.meta or None,
                        doc.score or None,
                        array.array("f", doc.embedding) if doc.embedding else None,
                    ]
                    if self._support_sparse_embeddings:
                        row.append(
                            oracledb.SparseVector(
                                cast(int, embedding_dim),
                                doc.sparse_embedding.indices,
                                array.array("f", doc.sparse_embedding.values),
                            )
                            if doc.sparse_embedding
                            else None
                        )
                    bind_input.append(tuple(row))

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

        if not self._support_sparse_embeddings and any(doc.sparse_embedding for doc in documents):
            raise ValueError("Sparse embeddings are not supported by this document store.")

        await self._ensure_initialized_async()
        embedding_dim = self._require_embedding_dim() if any(doc.sparse_embedding for doc in documents) else None

        async def context(
            connection: oracledb.AsyncConnection,
        ) -> int:
            with connection.cursor() as cursor:
                input_sizes = [None, None, None, oracledb.DB_TYPE_JSON, None, oracledb.DB_TYPE_JSON, None, None]
                if self._support_sparse_embeddings:
                    input_sizes.append(None)
                cursor.setinputsizes(*input_sizes)

                if policy == DuplicatePolicy.OVERWRITE:
                    query = _get_merge_query(
                        self._table_name, support_sparse_embeddings=self._support_sparse_embeddings
                    )

                else:
                    if policy == DuplicatePolicy.SKIP:
                        policy_hint = f"/*+ ignore_row_on_dupkey_index({self._table_name}(id)) */"
                    else:
                        policy_hint = ""
                    query = _get_insert_query(
                        self._table_name,
                        policy_hint,
                        support_sparse_embeddings=self._support_sparse_embeddings,
                    )

                bind_input = []
                for doc in documents:
                    row = [
                        doc.id,
                        doc.content or None,
                        doc.blob.data if doc.blob else None,
                        doc.blob.meta if doc.blob else None,
                        doc.blob.mime_type if doc.blob and doc.blob.mime_type else None,
                        doc.meta or None,
                        doc.score or None,
                        array.array("f", doc.embedding) if doc.embedding else None,
                    ]
                    if self._support_sparse_embeddings:
                        row.append(
                            oracledb.SparseVector(
                                cast(int, embedding_dim),
                                doc.sparse_embedding.indices,
                                array.array("f", doc.sparse_embedding.values),
                            )
                            if doc.sparse_embedding
                            else None
                        )
                    bind_input.append(tuple(row))

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
            if not self._support_sparse_embeddings:
                raise ValueError("Sparse embeddings are not supported by this document store.")
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
                if not self._support_sparse_embeddings:
                    raise ValueError("Sparse embeddings are not supported by this document store.")
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

    def _text_retrieval(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> list[Document]:
        self._ensure_initialized()

        params: dict[str, Any] = {"query": query}
        where_parts = ["CONTAINS(content, :query, 1) > 0"]
        if filters:
            bind_variables: list[Any] = []
            where_parts.append(_get_filter_string(filters, "meta", bind_variables))
            for i, value in enumerate(bind_variables):
                params[f"value{i}"] = value

        columns = [
            column
            for column in _get_document_columns(support_sparse_embeddings=self._support_sparse_embeddings)
            if column != "score"
        ]
        selected_columns = ", ".join([*columns, "SCORE(1) AS score"])
        db_query = f"""
            SELECT {selected_columns}
            FROM {self._table_name}
            WHERE {" AND ".join(where_parts)}
            ORDER BY SCORE(1) DESC
            FETCH FIRST {top_k} ROWS ONLY
            """

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                cursor.execute(db_query, params)
                result_columns = [col.name for col in cursor.description]
                results = self._get_result_to_documents(cursor.fetchall(), result_columns)

        return results

    async def _text_retrieval_async(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> list[Document]:
        await self._ensure_initialized_async()

        async def context(
            connection: oracledb.AsyncConnection,
        ) -> list[Document]:
            params: dict[str, Any] = {"query": query}
            where_parts = ["CONTAINS(content, :query, 1) > 0"]
            if filters:
                bind_variables: list[Any] = []
                where_parts.append(_get_filter_string(filters, "meta", bind_variables))
                for i, value in enumerate(bind_variables):
                    params[f"value{i}"] = value

            columns = [
                column
                for column in _get_document_columns(support_sparse_embeddings=self._support_sparse_embeddings)
                if column != "score"
            ]
            selected_columns = ", ".join([*columns, "SCORE(1) AS score"])
            db_query = f"""
                SELECT {selected_columns}
                FROM {self._table_name}
                WHERE {" AND ".join(where_parts)}
                ORDER BY SCORE(1) DESC
                FETCH FIRST {top_k} ROWS ONLY
                """

            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                await cursor.execute(db_query, params)
                result_columns = [col.name for col in cursor.description]
                results = self._get_result_to_documents(await cursor.fetchall(), result_columns)

            return results

        return await self._handle_context(context)

    async def _create_index_async(
        self, connection: oracledb.AsyncConnection, params: dict[str, Any] | None = None
    ) -> None:
        if params and "idx_name" in params:
            params["idx_name"] = _quote_identifier(params["idx_name"])

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

    async def _create_sparse_index_async(
        self,
        connection: oracledb.AsyncConnection,
        params: dict[str, Any] | None,
        distance_strategy: DistanceStrategy,
    ) -> None:
        if params and "idx_name" in params:
            params["idx_name"] = _quote_identifier(params["idx_name"])

        if params:
            if params["idx_type"] == "HNSW":
                await _create_hnsw_index_async(
                    connection, self._table_name, distance_strategy, params, "sparse_embedding"
                )
            elif params["idx_type"] == "IVF":
                await _create_ivf_index_async(
                    connection, self._table_name, distance_strategy, params, "sparse_embedding"
                )
            else:
                raise ValueError("Only supported indexes HNSW and IVF")
        else:
            await _create_hnsw_index_async(connection, self._table_name, distance_strategy, params, "sparse_embedding")

    def _create_index(self, connection: oracledb.Connection, params: dict[str, Any] | None = None) -> None:
        if params and "idx_name" in params:
            params["idx_name"] = _quote_identifier(params["idx_name"])

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

    def _create_sparse_index(
        self,
        connection: oracledb.Connection,
        params: dict[str, Any] | None,
        distance_strategy: DistanceStrategy,
    ) -> None:
        if params and "idx_name" in params:
            params["idx_name"] = _quote_identifier(params["idx_name"])

        if params:
            if params["idx_type"] == "HNSW":
                _create_hnsw_index(connection, self._table_name, distance_strategy, params, "sparse_embedding")
            elif params["idx_type"] == "IVF":
                _create_ivf_index(connection, self._table_name, distance_strategy, params, "sparse_embedding")
            else:
                raise ValueError("Only supported indexes HNSW and IVF")
        else:
            _create_hnsw_index(connection, self._table_name, distance_strategy, params, "sparse_embedding")

    @_handle_exceptions
    def create_hybrid_vector_index(
        self,
        idx_name: str,
        *,
        vectorizer_preference: OracleVectorizerPreference | None = None,
        text_embedder: Optional["OracleTextEmbedder"] = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if (vectorizer_preference is None) == (text_embedder is None):
            raise ValueError("Exactly one of 'vectorizer_preference' or 'text_embedder' must be provided.")

        self._ensure_initialized()

        should_drop_preference = False
        if text_embedder is not None:
            vectorizer_preference = OracleVectorizerPreference.create(self, text_embedder)
            should_drop_preference = True

        vectorizer_preference = cast(OracleVectorizerPreference, vectorizer_preference)
        try:
            quoted_idx_name = _quote_identifier(idx_name)
            ddl = _get_hybrid_index_ddl(self._table_name, quoted_idx_name, vectorizer_preference, params)

            with _get_connection(self._client) as connection:
                if not _index_exists(connection, quoted_idx_name, self._table_name):
                    with connection.cursor() as cursor:
                        cursor.execute(ddl)
        finally:
            if should_drop_preference:
                vectorizer_preference.drop()

    @_handle_exceptions_async
    async def create_hybrid_vector_index_async(
        self,
        idx_name: str,
        *,
        vectorizer_preference: OracleVectorizerPreference | None = None,
        text_embedder: Optional["OracleTextEmbedder"] = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if (vectorizer_preference is None) == (text_embedder is None):
            raise ValueError("Exactly one of 'vectorizer_preference' or 'text_embedder' must be provided.")

        await self._ensure_initialized_async()

        should_drop_preference = False
        if text_embedder is not None:
            vectorizer_preference = await OracleVectorizerPreference.create_async(self, text_embedder)
            should_drop_preference = True

        vectorizer_preference = cast(OracleVectorizerPreference, vectorizer_preference)
        try:
            quoted_idx_name = _quote_identifier(idx_name)
            ddl = _get_hybrid_index_ddl(self._table_name, quoted_idx_name, vectorizer_preference, params)

            async with _get_connection_async(self._client_async) as connection:
                if not await _index_exists_async(connection, quoted_idx_name, self._table_name):
                    with connection.cursor() as cursor:
                        await cursor.execute(ddl)
        finally:
            if should_drop_preference:
                await vectorizer_preference.drop_async()

    @_handle_exceptions
    def create_text_index(self, idx_name: str, *, column_name: str = "content") -> None:
        self._ensure_initialized()
        quoted_idx_name = _quote_identifier(idx_name)
        _validate_text_index_column(column_name)

        with _get_connection(self._client) as connection:
            _create_text_index(connection, self._table_name, quoted_idx_name, column_name)

    @_handle_exceptions_async
    async def create_text_index_async(self, idx_name: str, *, column_name: str = "content") -> None:
        await self._ensure_initialized_async()
        quoted_idx_name = _quote_identifier(idx_name)
        _validate_text_index_column(column_name)

        async with _get_connection_async(self._client_async) as connection:
            await _create_text_index_async(connection, self._table_name, quoted_idx_name, column_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        connection_params = data.get("init_parameters", {}).get("connection_params")
        if isinstance(connection_params, dict):
            _deserialize_connection_params(connection_params)
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_params=_serialize_connection_params(self._connection_params),
            table_name=self._table_name,
            use_connection_pool=self._use_connection_pool,
            embedding_dim=self._embedding_dim,
            support_sparse_embeddings=self._support_sparse_embeddings,
            create_vector_index=self._create_vector_index,
            vector_index_params=self._vector_index_params,
            vector_index_embedding_field=self._vector_index_embedding_field,
            vector_index_distance_strategy=self._vector_index_distance_strategy,
            sparse_vector_index=self._sparse_vector_index,
        )

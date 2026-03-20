import re
from typing import Any

from haystack.errors import FilterError

OPER_TYPES = {
    "==": (str, int, float),  # default operator (string, int, float)
    ">": (int, float),  # greater than (int, float)
    "<": (int, float),  # less than (int, float)
    "!=": (str, int, float),  # not equal to (string, int, float)
    ">=": (int, float),  # greater than or equal to (int, float)
    "<=": (int, float),  # less than or equal to (int, float)
    "in": (list,),  # In array (string or number)
    "not in": (list,),  # Not in array (string or number)
    "contains": (str, int, float, bool),  # Scalar contained in a metadata array
    "not contains": (str, int, float, bool),  # Scalar not contained in a metadata array
}

OPER_MAP = {
    "==": "{0} = {1}",  # default operator (string, int, float)
    ">": "{0} > {1}",  # greater than (int, float)
    "<": "{0} < {1}",  # less than (int, float)
    "!=": "{0} != {1}",  # not equal to (string, int, float)
    ">=": "{0} >= {1}",  # greater than or equal to (int, float)
    "<=": "{0} <= {1}",  # less than or equal to (int, float)
    "in": "{0} IN ({1})",  # In array (string or number)
    "not in": "{0} NOT IN ({1})",  # Not in array (string or number)
}


def _convert_oper_to_sql(oper: str, metadata_column: str, filter_key: str, value_bind: str) -> str:
    filter_key = ".".join(filter_key.split(".")[1:])

    if value_bind == "none":
        return (
            f"NOT JSON_EXISTS({metadata_column}, '$.{filter_key}') "
            f"OR JSON_EQUAL(JSON_QUERY({metadata_column}, '$.{filter_key}'), '[]') "
            f"OR JSON_EQUAL(JSON_QUERY({metadata_column}, '$.{filter_key}'), 'null')"
        )
    if value_bind == "not none":
        return (
            f"NOT (NOT JSON_EXISTS({metadata_column}, '$.{filter_key}') "
            f"OR JSON_EQUAL(JSON_QUERY({metadata_column}, '$.{filter_key}'), '[]') "
            f"OR JSON_EQUAL(JSON_QUERY({metadata_column}, '$.{filter_key}'), 'null'))"
        )
    elif oper == "contains":
        return f"JSON_EXISTS({metadata_column}, '$.{filter_key}[*]?(@ == $val)' PASSING {value_bind} AS \"val\")"
    elif oper == "not contains":
        return f"not JSON_EXISTS({metadata_column}, '$.{filter_key}[*]?(@ == $val)' PASSING {value_bind} AS \"val\")"
    else:
        if oper not in OPER_MAP:
            raise ValueError(f"Filter operation {oper} cannot be used with this vector store.")

        operation_f = OPER_MAP[oper]

        operation_s = operation_f.format(f"JSON_VALUE({metadata_column}, '$.{filter_key}')", value_bind)

        if oper in {"!=", "not in"}:
            return f"COALESCE({operation_s}, true)"
        else:
            return f"COALESCE({operation_s}, false)"


def _get_filter_string(filters: dict[str, Any], metadata_column: str, bind_variables: list[Any]) -> str:
    if filters.keys() != {"field", "value", "operator"} and filters.keys() != {"operator", "conditions"}:
        raise FilterError("Filter structure is not correct!")

    if "field" in filters:
        if not re.match(r"^[a-zA-Z0-9_.]+$", filters["field"]):
            raise FilterError(f"Invalid metadata key format: {filters['field']}")

        if filters["value"] and not isinstance(filters["value"], OPER_TYPES[filters["operator"]]):
            expected = OPER_TYPES[filters["operator"]]
            raise FilterError(
                f"Type for field {filters['field']} not correct; expected {expected}, got {type(filters['value'])}."
            )

        value_bind = ""
        if isinstance(filters["value"], list):
            # Needs multiple binds for a list https://python-oracledb.readthedocs.io/en/latest/user_guide/bind.html#binding-multiple-values-to-a-sql-where-in-clause
            value_binds = []
            for val in filters["value"]:
                value_binds.append(f":value{len(bind_variables)}")
                bind_variables.append(val)
            value_bind = ",".join(value_binds)
        elif filters["operator"] == "==" and filters["value"] is None:
            value_bind = "none"
        elif filters["operator"] == "!=" and filters["value"] is None:
            value_bind = "not none"
        else:
            value_bind = f":value{len(bind_variables)}"
            bind_variables.append(filters["value"])

        res = _convert_oper_to_sql(filters["operator"], metadata_column, filters["field"], value_bind)

        return res

    operator_upper = filters["operator"].upper()

    if operator_upper not in ["AND", "OR", "NOT"]:
        raise ValueError(f"Invalid operator: {filters['operator']}")

    # Combine all sub filters
    filter_strings = [_get_filter_string(f_, metadata_column, bind_variables) for f_ in filters["conditions"]]

    if operator_upper == "NOT":
        not_statement = f" {'AND'} ".join(filter_strings)
        return f" NOT ( {not_statement} )"
    else:
        return f" {operator_upper} ".join(filter_strings)

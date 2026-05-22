# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.components.document_stores.oracle.filters import _convert_oper_to_sql, _get_filter_string


def test_filter_contains_generates_json_exists_and_bind():
    bind_variables: list[object] = []

    result = _get_filter_string(
        {"field": "meta.tags", "operator": "contains", "value": "oracle"},
        "meta",
        bind_variables,
    )

    assert result == 'JSON_EXISTS(meta, \'$.tags[*]?(@ == $val)\' PASSING :value0 AS "val")'
    assert bind_variables == ["oracle"]


def test_filter_not_contains_generates_negated_json_exists_and_bind():
    bind_variables: list[object] = []

    result = _get_filter_string(
        {"field": "meta.tags", "operator": "not contains", "value": "oracle"},
        "meta",
        bind_variables,
    )

    assert result == 'not JSON_EXISTS(meta, \'$.tags[*]?(@ == $val)\' PASSING :value0 AS "val")'
    assert bind_variables == ["oracle"]


def test_filter_equal_none_generates_null_or_missing_check_without_binds():
    bind_variables: list[object] = []

    result = _get_filter_string(
        {"field": "meta.topic", "operator": "==", "value": None},
        "meta",
        bind_variables,
    )

    assert result == (
        "NOT JSON_EXISTS(meta, '$.topic') "
        "OR JSON_EQUAL(JSON_QUERY(meta, '$.topic'), '[]') "
        "OR JSON_EQUAL(JSON_QUERY(meta, '$.topic'), 'null')"
    )
    assert bind_variables == []


def test_filter_not_equal_none_generates_present_check_without_binds():
    bind_variables: list[object] = []

    result = _get_filter_string(
        {"field": "meta.topic", "operator": "!=", "value": None},
        "meta",
        bind_variables,
    )

    assert result == (
        "NOT (NOT JSON_EXISTS(meta, '$.topic') "
        "OR JSON_EQUAL(JSON_QUERY(meta, '$.topic'), '[]') "
        "OR JSON_EQUAL(JSON_QUERY(meta, '$.topic'), 'null'))"
    )
    assert bind_variables == []


def test_filter_nested_not_joins_inner_conditions():
    bind_variables: list[object] = []

    result = _get_filter_string(
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.number", "operator": ">", "value": 10},
                {"field": "meta.tags", "operator": "contains", "value": "oracle"},
            ],
        },
        "meta",
        bind_variables,
    )

    assert result == (
        " NOT ( COALESCE(JSON_VALUE(meta, '$.number') > :value0, false) "
        "AND JSON_EXISTS(meta, '$.tags[*]?(@ == $val)' PASSING :value1 AS \"val\") )"
    )
    assert bind_variables == [10, "oracle"]


def test_filter_not_in_and_or_and_validation_errors():
    bind_variables: list[object] = []
    result = _get_filter_string(
        {"field": "meta.topic", "operator": "not in", "value": ["oracle", "db"]},
        "meta",
        bind_variables,
    )
    assert result == "COALESCE(JSON_VALUE(meta, '$.topic') NOT IN (:value0,:value1), true)"
    assert bind_variables == ["oracle", "db"]

    bind_variables = []
    result = _get_filter_string(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.number", "operator": ">", "value": 10},
                {"field": "meta.topic", "operator": "==", "value": "oracle"},
            ],
        },
        "meta",
        bind_variables,
    )
    assert result == (
        "COALESCE(JSON_VALUE(meta, '$.number') > :value0, false) "
        "OR COALESCE(JSON_VALUE(meta, '$.topic') = :value1, false)"
    )
    assert bind_variables == [10, "oracle"]

    with pytest.raises(FilterError, match="Filter structure is not correct"):
        _get_filter_string({"field": "meta.topic"}, "meta", [])

    with pytest.raises(FilterError, match="expected"):
        _get_filter_string({"field": "meta.number", "operator": ">", "value": "ten"}, "meta", [])

    with pytest.raises(ValueError, match="Invalid operator"):
        _get_filter_string({"operator": "XOR", "conditions": []}, "meta", [])

    with pytest.raises(ValueError, match="cannot be used"):
        _convert_oper_to_sql("bad", "meta", "meta.topic", ":value0")


def test_filter_invalid_key_and_hybrid_type_error():
    with pytest.raises(FilterError, match="Invalid metadata key format"):
        _get_filter_string({"field": "meta.bad-key", "operator": "==", "value": "oracle"}, "meta", [])

    with pytest.raises(FilterError, match="string and numeric values"):
        from haystack_integrations.components.document_stores.oracle.filters import _infer_hybrid_filter_type

        _infer_hybrid_filter_type({"bad": "value"})

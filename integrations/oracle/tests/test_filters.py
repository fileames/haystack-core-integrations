# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.components.document_stores.oracle.filters import _get_filter_string


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

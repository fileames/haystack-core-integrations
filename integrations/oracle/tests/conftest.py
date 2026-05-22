# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

ORACLE_TEST_USER_ENV = "VECDB_USER"
ORACLE_TEST_PASSWORD_ENV = "VECDB_PASS"
ORACLE_TEST_DSN_ENV = "VECDB_HOST"

ORACLE_TESTS_CONFIGURED = all(
    os.getenv(var) for var in (ORACLE_TEST_USER_ENV, ORACLE_TEST_PASSWORD_ENV, ORACLE_TEST_DSN_ENV)
)
ORACLE_TESTS_REASON = (
    f"Set {ORACLE_TEST_USER_ENV}, {ORACLE_TEST_PASSWORD_ENV}, and {ORACLE_TEST_DSN_ENV} to run Oracle integration tests."
)


def oracle_test_connection_params() -> dict[str, str]:
    user = os.getenv(ORACLE_TEST_USER_ENV)
    password = os.getenv(ORACLE_TEST_PASSWORD_ENV)
    dsn = os.getenv(ORACLE_TEST_DSN_ENV)

    if not (user and password and dsn):
        raise RuntimeError(ORACLE_TESTS_REASON)

    return {"user": user, "password": password, "dsn": dsn}


def oracle_test_connect_dsn() -> str:
    params = oracle_test_connection_params()
    return f'{params["user"]}/{params["password"]}@{params["dsn"]}'


def oracle_unit_test_connection_params() -> dict[str, str]:
    return {
        "user": os.getenv(ORACLE_TEST_USER_ENV, ""),
        "password": os.getenv(ORACLE_TEST_PASSWORD_ENV, ""),
        "dsn": os.getenv(ORACLE_TEST_DSN_ENV, ""),
    }

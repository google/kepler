# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to simplify unit tests.

For simplicity, these tests assume a database called 'test' already exists.

"""

import time
import psycopg2

from kepler.training_data_collection_pipeline import query_utils

TEST_QUERY_ID = 'q0_0'
TEST_TEMPLATE_STRING = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo JOIN bar on x = b WHERE \\n a < @param0 \\n and bar.c = '@param1';",
  "predicates": [
    {
      "alias": "bar",
      "column": "c",
      "operator": "="
    }
  ]
}
"""

HINTS_0 = """{"hints": "/*+ MergeJoin(foo bar) Leading((foo bar)) */", "source": "default"}"""
HINTS_1 = """{"hints": "/*+ HashJoin(foo bar) Leading((foo bar)) */", "source": "default"}"""

_DBNAME = 'test'
USER = 'test'
PASSWORD = 'test'

_SCHEMA = """
CREATE TABLE foo(x int, y int);
CREATE TABLE bar(a int, b int, c varchar, d_date date, e_date timestamp, website_url varchar);
CREATE TABLE baz(j int, k int, l varchar);
"""

_DATA_FOO = [
    [1, -1],
    [1, 2],
    [1, 3],
    [2, 2],
    [2, 4],
]

_DATA_BAR = [
    [
        1, 1, 'alfa', '2021-09-19', '2022-02-07 14:28:59.473387-08',
        'https://hello.com'
    ],
    [
        1, 2, 'bravo', '2021-09-21', '2022-02-08 14:28:59.473387-08',
        'http://goodbye.org/methods'
    ],
    [
        1, 3, 'bravo', '2021-09-22', '2022-02-07 15:28:59.473387-08',
        'http://www.goodnight.org'
    ],
    [
        3, 2, 'charlie', '2021-09-20', '2022-02-07 13:28:59.473387-08',
        'http://www.goodmorning.com'
    ],
    [3, 2, None, None, None, None],
]

_DATA_BAZ = [
    [1, 3, 'single-string'],
    [1, 4, 'split string'],
    [2, 4, 'single-string'],
    [3, 5, 'split string'],
]

_TABLE_LIST = ['foo', 'bar', 'baz']


def populate_database(query_manager: query_utils.QueryManager):
  """Create schema and insert data for test cases."""
  query_manager.execute_and_commit(_SCHEMA)

  columns_list = ['x, y', 'a, b, c, d_date, e_date, website_url', 'j, k, l']
  data_list = [_DATA_FOO, _DATA_BAR, _DATA_BAZ]

  for table, columns, data in zip(_TABLE_LIST, columns_list, data_list):
    for row in data:
      row_values_as_strings = [
          f"'{str(value)}'" if value else 'NULL' for value in row
      ]
      query_manager.execute_and_commit(
          f"INSERT INTO {table} ({columns}) VALUES ({','.join(row_values_as_strings)})"
      )


class TestDatabase:
  """Manages the creating and dropping of a test-case-specific database."""

  def __init__(self):
    """Creates test database.

    Connects to the preexisting test database to create a new database with a
    unique name.

    Attributes:
      dbname: The name of the newly created database. This name will be unique
        each time a TestDatabase is created.
    """
    self.dbname = f'test_{time.time_ns()}'

    conn = psycopg2.connect(
        dbname=_DBNAME, user=USER, password=PASSWORD, host='localhost')
    # Enabling AUTOCOMMIT is required to execute CREATE DATABASE.
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    self._cursor = conn.cursor()
    self._cursor.execute(f'CREATE DATABASE {self.dbname};')

    # Complete setup of the newly created database per README instructions.
    query_manager = query_utils.QueryManager(
        query_utils.DatabaseConfiguration(
            dbname=self.dbname, user=USER, password=PASSWORD))
    query_manager.execute_and_commit('CREATE EXTENSION pg_stat_statements;')

  def drop(self):
    self._cursor.execute('SELECT pg_terminate_backend(pg_stat_activity.pid) '
                         'FROM pg_stat_activity '
                         f"WHERE pg_stat_activity.datname = '{self.dbname}';")
    self._cursor.execute(f'DROP DATABASE {self.dbname};')

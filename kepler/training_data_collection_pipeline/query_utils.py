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

"""Provides query access to the database.

The QueryManager abstracts away communication with the database. When provided
queries include parameters, they are expected to be in the format of @param#
numbered from 0 to n.

```
Typical usage:

query_manager = QueryManager()
query_manager.connect_postgres(database_name)
query_manager.execute(sql_text)
```
"""
import dataclasses
import enum
import json
import os
from typing import Any, List, Optional, Sequence, Tuple

from absl import logging
import psycopg2
import psycopg2.errorcodes

from kepler.training_data_collection_pipeline import utils

_GET_INDEXES_QUERY = """
SELECT
    tablename,
    indexname,
    indexdef
FROM
    pg_indexes
WHERE
    schemaname = 'public'
ORDER BY
    tablename,
    indexname;
"""

# From https://www.postgresql.org/docs/13/runtime-config-query.html.
_POSTGRES_COST_CONSTANTS = [
    'seq_page_cost', 'random_page_cost', 'cpu_tuple_cost',
    'cpu_index_tuple_cost', 'cpu_operator_cost', 'parallel_setup_cost',
    'parallel_tuple_cost', 'min_parallel_table_scan_size',
    'min_parallel_index_scan_size', 'effective_cache_size', 'jit_above_cost',
    'jit_inline_above_cost', 'jit_optimize_above_cost'
]

# From https://www.postgresql.org/docs/13/runtime-config-resource.html.
# Currently only including configs from first three sections.
_POSTGRES_RESOURCE_CONFIGS = [
    'shared_buffers', 'huge_pages', 'temp_buffers', 'max_prepared_transactions',
    'work_mem', 'hash_mem_multiplier', 'maintenance_work_mem',
    'autovacuum_work_mem', 'max_stack_depth', 'shared_memory_type',
    'dynamic_shared_memory_type', 'temp_file_limit', 'max_files_per_process'
]

JSON = Any


class PostgresDataType(enum.Enum):
  INTEGER = 'integer'
  VARCHAR = 'character varying'
  DATE = 'date'
  TIMESTAMP = 'timestamp without time zone'


@dataclasses.dataclass
class DatabaseConfiguration:
  dbname: str
  user: Optional[str] = None
  password: Optional[str] = None
  host: Optional[str] = 'localhost'
  seed: Optional[float] = 0


# TODO(lyric): Make a dataclass to represent connection information, eg dbname,
# user, password. Refactor usage through to simplify function signatures that
# accept the connection information as 4 or 5 arguments.
class QueryManager:
  """QueryManager wraps query, DDL, and DML access to a database.

  Attributes:
    database_configuration: The configuration describing the database
      connection.
  """

  def __init__(self, database_configuration: DatabaseConfiguration):
    """Connects to a Postgres database using the psycopg2 client.

    Args:
      database_configuration: The configuration describing the database
        connection.
    """
    self.database_configuration = database_configuration

    self._conn = psycopg2.connect(
        dbname=self.database_configuration.dbname,
        user=self.database_configuration.user,
        password=self.database_configuration.password,
        host=self.database_configuration.host)
    self._cursor = self._conn.cursor()
    self._cursor.execute(f'SELECT SETSEED({self.database_configuration.seed})')
    self._cursor.execute("LOAD 'pg_hint_plan'")

  def refresh_cursor(self) -> None:
    self._cursor.close()
    self._cursor = self._conn.cursor()

  def enable_pg_hint_plan_debug(self) -> None:
    self._cursor.execute('set pg_hint_plan.debug_print to detailed;')

  def disable_pg_hint_plan_debug(self) -> None:
    self._cursor.execute('set pg_hint_plan.debug_print to off;')

  def run_analyze(self) -> None:
    self._cursor.execute('analyze;')

  def get_postgres_config_info(self) -> JSON:
    return {
        'indexes': self.get_index_info(),
        'cost_constants': self.get_cost_constants(),
        'resource_configs': self.get_resource_configs()
    }

  def get_index_info(self) -> JSON:
    self._cursor.execute(_GET_INDEXES_QUERY)
    return self._cursor.fetchall()

  def get_cost_constants(self) -> JSON:
    return {
        config: self.execute(f'SHOW {config};')
        for config in _POSTGRES_COST_CONSTANTS
    }

  def get_resource_configs(self) -> JSON:
    return {
        config: self.execute(f'SHOW {config};')
        for config in _POSTGRES_RESOURCE_CONFIGS
    }

  def execute_and_commit(self, sql: str) -> None:
    """Executes and commits the provided sql.

    Args:
      sql: The SQL text. The expected usage is DDL and DML.
    """
    self._cursor.execute(sql)
    self._conn.commit()

  def execute_timed(
      self,
      query: str,
      params: Optional[Sequence[Any]] = None,
      timeout_ms: Optional[float] = None
  ) -> Tuple[Optional[float], Optional[int]]:
    """Returns the execution time in ms and rows produced for the provided query.

    This implementation relies on there being exactly one query being executed
    at a time.

    Args:
      query: SQL query template string with 0 or more parameters provided in the
        form of @param#, starting with 0.
      params: List of parameter values to substitute into the query. All values
        will be cast to str.
      timeout_ms: The statement timeout for this query in ms.

    Returns:
      Tuple:
        1) The execution time of the query in ms or None if the query times out.
        2) The number of rows produced by the query or None if the query times
          out.
    """
    self.refresh_cursor()
    executable_query = utils.substitute_query_params(query, params)

    if timeout_ms:
      self._cursor.execute(f"SET statement_timeout TO '{timeout_ms}'")

    self._cursor.execute('select pg_stat_statements_reset();')

    result = None

    rows = 0
    try:
      self._cursor.execute(executable_query)

      # Fetch all the results without loading them all into memory at once.
      while True:
        rows_fetched = self._cursor.fetchmany(10000)
        if not rows_fetched:
          break
        rows += len(rows_fetched)

      # We don't want the timeout to affect gathering stats about query
      # execution.
      if timeout_ms:
        self._cursor.execute('SET statement_timeout TO 0')

      self._cursor.execute(
          "SELECT total_exec_time from pg_stat_statements where query != 'BEGIN' "
          "and query not like '%pg_stat_statements_reset%' "
          "and query not like '%SET statement_timeout TO%' and calls = 1;")

      result = self._cursor.fetchone()
      assert len(result) == 1
      result = result[0]
      assert isinstance(result, float)

    except psycopg2.OperationalError as e:
      assert e.pgcode == psycopg2.errorcodes.QUERY_CANCELED
      # We must END the aborted transaction before running any more queries.
      self._cursor.execute('END')

      # Run this again in case it was skipped during the try.
      if timeout_ms:
        self._cursor.execute('SET statement_timeout TO 0')

    return result, rows if result else None

  def execute(self,
              query: str,
              params: Optional[Sequence[Any]] = None) -> List[Tuple[Any]]:
    """Executes the provided SELECT query after substituting params.

    Args:
      query: SQL query template string with 0 or more parameters provided in the
        form of @param#, starting with 0.
      params: List of parameter values to substitute into the query. All values
        will be cast to str.

    Returns:
      A list of tuples in which each tuple represents a row from the result set.
    """
    self.refresh_cursor()
    executable_query = utils.substitute_query_params(query, params)
    logging.debug(executable_query)

    self._cursor.execute(executable_query)
    return self._cursor.fetchall()

  def get_query_plan_and_execute(
      self,
      query: str,
      params: Optional[Sequence[Any]] = None,
      configuration_parameters: Optional[Sequence[str]] = None) -> Any:
    """Runs EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) for a specific query and params.

    VERBOSE is omitted since the information isn't useful.
    Any provided configuration_parameters are SET to OFF for the retrieval of
    the plan.

    Args:
      query: SQL query template string with 0 or more parameters provided in the
        form of @param#, starting with 0.
      params: List of parameter values to substitute into the query. All values
        will be cast to str.
      configuration_parameters: List of Postgres configuration parameters, such
        as `enable_nestloop`, to toggle off when retrieving the EXPLAIN plan.

    Returns:
      The results of the EXPLAIN ANALYZE, BUFFERS command in JSON format.
    """
    executable_query = utils.substitute_query_params(query, params)
    query_string = f'EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {executable_query}'
    return self._execute_query_with_configs(query_string,
                                            configuration_parameters)

  # Using Any as catch-all for the returned JSON object.
  def get_query_plan(
      self,
      query: str,
      params: Optional[Sequence[Any]] = None,
      configuration_parameters: Optional[List[str]] = None) -> Any:
    """Retrieves the EXPLAIN plan for a SELECT query after substituting params.

    Any provided configuration_parameters are SET to OFF for the retrieval of
    the plan.

    Args:
      query: SQL query template string with 0 or more parameters provided in the
        form of @param#, starting with 0.
      params: List of parameter values to substitute into the query. All values
        will be cast to str.
      configuration_parameters: List of Postgres configuration parameters, such
        as `enable_nestloop`, to toggle off when retrieving the EXPLAIN plan.

    Returns:
      The EXPLAIN plan in JSON format.
    """
    executable_query = utils.substitute_query_params(query, params)
    query_string = f'EXPLAIN (FORMAT JSON) {executable_query}'
    return self._execute_query_with_configs(query_string,
                                            configuration_parameters)

  def _execute_query_with_configs(
      self,
      query_string: str,
      configuration_parameters: Optional[Sequence[str]] = None) -> Any:
    """Executes a query, optionally with Postgres optimizer configuration parameters.

    Args:
      query_string: Full query represented as a string.
      configuration_parameters: List of Postgres configuration parameters, such
        as `enable_nestloop`, to toggle off when retrieving the EXPLAIN plan.

    Returns:
      The result of the query.
    """
    self.refresh_cursor()
    logging.debug(query_string)
    configuration_parameters_list = configuration_parameters if configuration_parameters else []

    for configuration_parameter in configuration_parameters_list:
      self._cursor.execute(f'SET {configuration_parameter} TO OFF')

    self._cursor.execute(query_string)
    return_value = self._cursor.fetchone()[0][0]

    for configuration_parameter in configuration_parameters_list:
      self._cursor.execute(f'SET {configuration_parameter} TO ON')

    self._cursor.execute('END')
    return return_value

  def get_column_type(self, table: str, column: str) -> PostgresDataType:
    """Gets the Postgres type of the specified column.

    Args:
      table: Name of table.
      column: Name of column.

    Returns:
      The Postgres type of the column.
    """
    self.refresh_cursor()
    query_string = f"""SELECT data_type FROM information_schema.columns
                       WHERE table_name=\'{table}\'
                       AND column_name=\'{column}\'"""
    self._cursor.execute(query_string)
    return PostgresDataType(self._cursor.fetchone()[0])

  def get_distinct_values(self, table: str, column: str) -> List[Any]:
    """Retrieves all distinct values in a specified column.

    Returns the distinct values in descending order based on frequency.

    Args:
      table: Name of table.
      column: Name of column.

    Returns:
      All distinct values in the column in descending order of frequency.
    """
    self.refresh_cursor()
    query_string = f"""SELECT {column} FROM {table}
                       GROUP BY {column} ORDER BY COUNT(*) DESC"""
    self._cursor.execute(query_string)
    return [x[0] for x in self._cursor.fetchall()]

  def get_numeric_column_statistics(self,
                                    table: str,
                                    column: str) -> Tuple[Optional[float],
                                                          Optional[float]]:
    """Gets mean and variance of the values in a numeric column.

    Args:
      table: Name of table.
      column: Name of column.

    Returns:
      Tuple consisting of mean and variance. Returns (None, None) if the
        column is not numeric (e.g. varchar).
    """
    self.refresh_cursor()
    query_string = f'SELECT AVG({column}), VARIANCE({column}) FROM {table}'
    try:
      self._cursor.execute(query_string)
    except psycopg2.Error as e:
      assert e.pgcode == psycopg2.errorcodes.UNDEFINED_FUNCTION
      return None, None
    stats = self._cursor.fetchone()

    # psycopg2 returns decimal.Decimal objects.
    return tuple(map(float, stats))

  def get_column_bounds(self,
                        table: str,
                        column: str) -> Tuple[Optional[int], Optional[int]]:
    """Gets min and max of the values in a column.

    Args:
      table: Name of table.
      column: Name of column.

    Returns:
      Tuple consisting of the min and max.
    """
    self.refresh_cursor()
    query_string = f'SELECT MIN({column}), MAX({column}) FROM {table}'
    self._cursor.execute(query_string)
    stats = self._cursor.fetchone()

    return stats

  def _get_pg_stats_data(self,
                         pg_stats_column: str,
                         table: str,
                         column: str) -> Any:
    """Retrieves a pg_stats column for a specified table and column.

    This function assumes that there is only a single row in the corresponding
    pg_stats column for the given table and column.

    Args:
      pg_stats_column: Column of pg_stats to fetch.
      table: Name of table.
      column: Name of column.

    Returns:
      The first row of the corresponding pg_stats column.
    """
    self.refresh_cursor()
    query_string = f"""SELECT {pg_stats_column} FROM pg_stats
                       WHERE tablename=\'{table}\' AND attname=\'{column}\'"""
    self._cursor.execute(query_string)
    return self._cursor.fetchone()[0]

  def get_most_common_values(self, table: str, column: str) -> List[str]:
    """Retrieves most common values in pg_stats for a specified table and column.

    Args:
      table: Name of table.
      column: Name of column.

    Returns:
      The contents of the corresponding pg_stats column.
    """
    values = self._get_pg_stats_data('most_common_vals', table, column)
    if not values:
      return []
    values = values.strip('{}').split(',')
    # Strings with a space have double quotes around them.
    return list(map(lambda x: x.strip('\"'), values))

  def get_most_common_frequencies(self, table: str, column: str) -> List[float]:
    """Retrieves most_common_freqs in pg_stats for a specified table and column.

    Args:
      table: Name of table.
      column: Name of column.

    Returns:
      The contents of the corresponding pg_stats column.
    """
    values = self._get_pg_stats_data('most_common_freqs', table, column)
    if not values:
      return []
    return list(map(float, values))


def save_postgres_config_info(query_manager: QueryManager,
                              base_output_dir: str) -> None:
  """Saves Postgres configuration info in JSON format.

  Args:
    query_manager: The QueryManager object.
    base_output_dir: Filepath to the base output directory. This function will
      create a new folder 'pg_configs' under this base directory, and the output
      file will be 'pg_configs/pg_configs.json'.
  """
  pg_config_dir = os.path.join(base_output_dir, 'pg_configs')
  os.makedirs(pg_config_dir, exist_ok=True)

  pg_config_info = query_manager.get_postgres_config_info()
  with open(os.path.join(pg_config_dir, 'pg_configs.json'), 'w') as f:
    json.dump(pg_config_info, f)

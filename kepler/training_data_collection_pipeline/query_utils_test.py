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

"""Tests for QueryManager.

For simplicity, we currently presume the existence of a database named test.
"""
from typing import Any, List
from absl.testing import absltest
from absl.testing import parameterized

from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util

_TEST_QUERY_0 = "SELECT x, a FROM foo JOIN BAR on x = a WHERE y > 2"


def _verify_explain_plan_contains_helper(node: Any, key: str,
                                         value: str) -> bool:
  if key in node and node[key] == value:
    return True

  return any(
      _verify_explain_plan_contains_helper(child, key, value)
      for child in node["Plans"]) if "Plans" in node else False


def _verify_explain_plan_contains(plan: Any, key: str, value: str) -> bool:
  """Walks tree and returns true if at least one key has the provided value."""
  return _verify_explain_plan_contains_helper(plan["Plan"], key, value)


def _get_index_name(table: str, columns: List[str]) -> str:
  return "_".join([table] + columns + ["idx"])


class QueryManagerPostgresTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._query_manager = query_utils.QueryManager(
        query_utils.DatabaseConfiguration(
            dbname=self._test_database.dbname,
            user=test_util.USER,
            password=test_util.PASSWORD))
    test_util.populate_database(self._query_manager)
    # Some tests read from pg_stats tables, which requires running ANALYZE.
    # Warning: Removing/adding this affects cardinality estimates.
    self._query_manager.run_analyze()

  def tearDown(self):
    self._test_database.drop()
    super().tearDown()

  def _create_index(self, table: str, columns: List[str]) -> None:
    index_name = _get_index_name(table, columns)
    index_cols = ", ".join(columns)
    self._query_manager.execute_and_commit(
        f"CREATE INDEX {index_name} ON {table} USING btree ({index_cols});")

  def _drop_index(self, table: str, columns: List[str]) -> None:
    index_name = _get_index_name(table, columns)
    self._query_manager.execute_and_commit(f"DROP INDEX {index_name};")

  def test_get_indexes_info(self):
    indexes = [("bar", ["c", "d_date", "website_url"]), ("baz", ["k"]),
               ("foo", ["x"]), ("foo", ["x", "y"])]
    for table, columns in indexes:
      self._create_index(table, columns)
    index_data = self._query_manager.get_index_info()

    for i in range(len(index_data)):
      table, columns = indexes[i]
      target_index_name = _get_index_name(table, columns)
      self.assertEqual(index_data[i][0], table)
      self.assertEqual(index_data[i][1], target_index_name)

    for table, columns in indexes:
      self._drop_index(table, columns)

  def test_get_cost_constants(self):
    cost_constants = self._query_manager.get_cost_constants()
    self.assertIn("random_page_cost", cost_constants)
    self.assertLen(cost_constants["random_page_cost"], 1)

  def test_get_resource_configs(self):
    resource_configs = self._query_manager.get_resource_configs()
    self.assertIn("shared_buffers", resource_configs)
    self.assertLen(resource_configs["shared_buffers"], 1)

  def test_execute(self):
    """Verifies simple query execution with and without params."""
    expected = [(1, 3), (2, 4)]

    results = self._query_manager.execute("SELECT x, y FROM foo where y > 2")
    self.assertEqual(results, expected)

    results = self._query_manager.execute(
        "SELECT x, y FROM foo where y > @param0", [2])
    self.assertEqual(results, expected)

  def test_get_query_plan(self):
    """Verifies retrieving an EXPLAIN plan with and without params."""

    plan_0 = self._query_manager.get_query_plan(_TEST_QUERY_0)
    self.assertTrue(_verify_explain_plan_contains(plan_0, "Filter", "(y > 2)"))

    plan_1 = self._query_manager.get_query_plan(
        "SELECT x, a FROM foo JOIN BAR on x = a where y > @param0", [2])
    self.assertTrue(_verify_explain_plan_contains(plan_0, "Filter", "(y > 2)"))

    self.assertEqual(plan_0, plan_1)

  def test_get_query_plan_toggle_config_params(self):
    """Verifies that provided configuration parameters are disabled for the plan.

    The configuration parameters should also be re-enabled after retrieving the
    plan and should not affect subsequent calls.

    """
    plan_0 = self._query_manager.get_query_plan(_TEST_QUERY_0)
    self.assertTrue(
        _verify_explain_plan_contains(plan_0, "Node Type", "Hash Join"))
    self.assertFalse(
        _verify_explain_plan_contains(plan_0, "Node Type", "Merge Join"))

    plan_1 = self._query_manager.get_query_plan(
        _TEST_QUERY_0, configuration_parameters=["enable_hashjoin"])
    self.assertFalse(
        _verify_explain_plan_contains(plan_1, "Node Type", "Hash Join"))
    self.assertTrue(
        _verify_explain_plan_contains(plan_1, "Node Type", "Nested Loop"))

    plan_2 = self._query_manager.get_query_plan(_TEST_QUERY_0)
    self.assertTrue(
        _verify_explain_plan_contains(plan_2, "Node Type", "Hash Join"))
    self.assertFalse(
        _verify_explain_plan_contains(plan_2, "Node Type", "Merge Join"))

    self.assertEqual(plan_0, plan_2)

  def test_execute_timed(self):
    """Verifies retrieving execution latency for a query."""
    latency, rows = self._query_manager.execute_timed(
        "SELECT x, y FROM foo where y > 2")
    self.assertIsNotNone(latency)
    self.assertIsInstance(latency, float)
    self.assertGreater(latency, 0)
    self.assertEqual(rows, 2)

  def test_execute_timed_local(self):
    """Verifies retrieving execution latency for a query."""
    latency, rows = self._query_manager.execute_timed_local(
        "SELECT x, y FROM foo where y > 2"
    )
    self.assertIsNotNone(latency)
    self.assertIsInstance(latency, float)
    self.assertGreater(latency, 0)
    self.assertEqual(rows, 2)

  def test_execute_timed_no_rows(self):
    """Verifies retrieving execution latency for a query."""
    latency, rows = self._query_manager.execute_timed(
        "SELECT x, y FROM foo where y > 2 and y < 1")
    self.assertIsNotNone(latency)
    self.assertIsInstance(latency, float)
    self.assertGreater(latency, 0)
    self.assertEqual(rows, 0)

  def test_execute_timed_local_no_rows(self):
    """Verifies retrieving execution latency for a query."""
    latency, rows = self._query_manager.execute_timed_local(
        "SELECT x, y FROM foo where y > 2 and y < 1"
    )
    self.assertIsNotNone(latency)
    self.assertIsInstance(latency, float)
    self.assertGreater(latency, 0)
    self.assertEqual(rows, 0)

  def test_execute_timed_timeout(self):
    """Verifies enforcement of query timeout."""
    latency, rows = self._query_manager.execute_timed(
        "SELECT pg_sleep(1)", timeout_ms=1)
    self.assertIsNone(latency)
    self.assertIsNone(rows)

    # Check that the timeout has been reset.
    latency, rows = self._query_manager.execute_timed("SELECT pg_sleep(1)")
    self.assertIsNotNone(latency)
    self.assertGreater(latency, 0)
    self.assertEqual(rows, 1)

  def test_execute_timed_local_timeout(self):
    """Verifies enforcement of query timeout."""
    latency, rows = self._query_manager.execute_timed_local(
        "SELECT pg_sleep(1)", timeout_ms=1
    )
    self.assertIsNone(latency)
    self.assertIsNone(rows)

    # Check that the timeout has been reset.
    latency, rows = self._query_manager.execute_timed("SELECT pg_sleep(1)")
    self.assertIsNotNone(latency)
    self.assertGreater(latency, 0)
    self.assertEqual(rows, 1)

  def test_get_query_plan_and_execute(self):
    """Verifies running EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) for a query."""
    output = self._query_manager.get_query_plan_and_execute(_TEST_QUERY_0)
    explain_analyze_keys = [
        "Plan", "Planning", "Planning Time", "Triggers", "Execution Time"
    ]
    for key in explain_analyze_keys:
      self.assertIn(key, output)

    # Check that some essential subset of keys are present within each node.
    plan_keys = [
        "Node Type", "Total Cost", "Plan Rows", "Plan Width",
        "Actual Total Time", "Actual Rows", "Actual Loops", "Shared Hit Blocks",
        "Shared Read Blocks", "Local Hit Blocks", "Local Read Blocks"
    ]
    for key in plan_keys:
      self.assertIn(key, output["Plan"])

  def test_set_seed(self):
    """Verifies setting the random number seed."""
    random_0 = self._query_manager.execute("SELECT random()")[0][0]
    self.assertAlmostEqual(random_0, 3.907985046680551e-14, delta=1e-24)

    query_manager = query_utils.QueryManager(
        query_utils.DatabaseConfiguration(
            dbname=self._test_database.dbname,
            user=test_util.USER,
            password=test_util.PASSWORD,
            seed=1))
    random_1 = query_manager.execute("SELECT random()")[0][0]
    self.assertAlmostEqual(random_1, 0.4999104186659835, delta=1e-10)

  @parameterized.named_parameters(
      ("foo_x", "foo", "x", query_utils.PostgresDataType.INTEGER),
      ("bar_c", "bar", "c", query_utils.PostgresDataType.VARCHAR),
      ("bar_d_date", "bar", "d_date", query_utils.PostgresDataType.DATE),
      ("bar_e_date", "bar", "e_date", query_utils.PostgresDataType.TIMESTAMP),
      ("baz_l", "baz", "l", query_utils.PostgresDataType.VARCHAR),
  )
  def test_get_column_type(self, table, column, expected_type):
    data_type = self._query_manager.get_column_type(table, column)
    self.assertEqual(data_type, expected_type)

  @parameterized.named_parameters(
      ("foo_x", "foo", "x", [1, 2]),
      ("bar_c", "bar", "c", ["bravo", None, "alfa", "charlie"]),
      ("baz_j", "baz", "j", [1, 2, 3]),
      ("baz_l", "baz", "l", ["single-string", "split string"]),
  )
  def test_get_distinct_values(self, table, column, expected_distinct):
    distinct_values = self._query_manager.get_distinct_values(table, column)
    self.assertEqual(distinct_values, expected_distinct)

  @parameterized.named_parameters(
      ("foo_x", "foo", "x", (1.4, 0.3)),
      ("foo_y", "foo", "y", (2., 3.5)),
      ("bar_a", "bar", "a", (1.8, 1.2)),
      ("bar_b", "bar", "b", (2., 0.5)),
      ("baz_l", "baz", "l", (None, None)),
      ("baz_length_l", "baz", "length(l)", (12.5, 1/3)),
  )
  def test_get_numeric_column_statistics(self, table, column, expected_stats):
    stats = self._query_manager.get_numeric_column_statistics(table, column)
    self.assertEqual(stats, expected_stats)

  @parameterized.named_parameters(
      ("foo_x", "foo", "x", (1, 2)),
      ("foo_y", "foo", "y", (-1, 4)),
      ("bar_a", "bar", "a", (1, 3)),
      ("bar_b", "bar", "b", (1., 3)),
      ("bar_c", "bar", "c", ("alfa", "charlie")),
      ("baz_l", "baz", "l", ("single-string", "split string")),
      ("baz_length_l", "baz", "length(l)", (12, 13)),
  )
  def test_get_column_bounds(self, table, column, expected_bounds):
    stats = self._query_manager.get_column_bounds(table, column)
    self.assertEqual(stats, expected_bounds)

  @parameterized.named_parameters(
      ("foo_x", "foo", "x", ["1", "2"]),
      ("bar_c", "bar", "c", ["bravo"]),
      ("bar_d_date", "bar", "d_date", []),
      ("baz_l", "baz", "l", ["single-string", "split string"]),
  )
  def test_get_most_common_values(self, table, column, expected_mcvs):
    mcvs = self._query_manager.get_most_common_values(table, column)
    self.assertEqual(mcvs, expected_mcvs)

  @parameterized.named_parameters(
      ("foo_x", "foo", "x", [0.6, 0.4]),
      ("bar_c", "bar", "c", [0.4]),
      ("baz_l", "baz", "l", [0.5, 0.5]),
  )
  def test_get_most_common_frequencies(self, table, column, expected_mcfs):
    mcfs = self._query_manager.get_most_common_frequencies(table, column)
    self.assertEqual(mcfs, expected_mcfs)

if __name__ == "__main__":
  absltest.main()

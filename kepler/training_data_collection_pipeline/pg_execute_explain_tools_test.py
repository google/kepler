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

"""Tests for pg_execute_explain_tools.py."""

import json
from typing import Any, List, Optional, Set

from kepler.training_data_collection_pipeline import pg_execute_explain_tools
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util

from absl.testing import absltest
from absl.testing import parameterized

# Typing aliases.
JSON = Any

_TEST_QUERY_0 = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo JOIN bar on x = b \\n JOIN baz on x = j WHERE \\n bar.a < @param0 \\n and bar.c = '@param1' \\n order by y;"
}
"""

_TEST_QUERY_1 = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo JOIN bar on x = b \\n JOIN baz on x = j WHERE \\n bar.a < @param0 \\n and bar.c = '@param1' \\n and foo.x is null \\n order by y;"
}
"""

_TEST_QUERY_NO_JOINS = """
{
  "query": "SELECT a, b \\n FROM \\n bar WHERE \\n bar.a < @param0 \\n and bar.c = '@param1';"
}
"""

_TEST_QUERY_JOINS_1 = """
{
  "query": "SELECT COUNT(*) FROM foo as foo_a, foo as foo_b WHERE foo_a.x < @param0 \\n AND foo_a.x < LENGTH('@param1');"
}
"""

_TEST_QUERY_JOINS_2 = """
{
  "query": "SELECT COUNT(*) FROM foo as foo_a, foo as foo_b, foo as foo_c WHERE foo_a.x < @param0 \\n AND foo_a.x < LENGTH('@param1');"
}
"""

_TEST_QUERY_JOINS_3 = """
{
  "query": "SELECT COUNT(*) FROM foo as foo_a, foo as foo_b, foo as foo_c, foo as foo_d WHERE foo_a.x < @param0 \\n AND foo_a.x < LENGTH('@param1');"
}
"""

_VALUES = """
{
  "params": [[0, "alpha"], [1, "bravo"], [1, "charlie"]]
}
"""

_HINTS = """
[
  {"hints": "/*+ MergeJoin(foo bar baz) MergeJoin(foo bar) Leading((baz (foo bar))) */", "source": "default"},
  {"hints": "/*+ NestLoop(foo baz bar) HashJoin(baz bar) Leading((foo (baz bar))) */", "source": "default"}
]
"""


class PgExecuteExplainToolsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._database_configuration = query_utils.DatabaseConfiguration(
        dbname=self._test_database.dbname,
        user=test_util.USER,
        password=test_util.PASSWORD)
    self._query_manager = query_utils.QueryManager(self._database_configuration)
    test_util.populate_database(self._query_manager)

    self._query_id = test_util.TEST_QUERY_ID
    self._hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS)}
    self._values = {test_util.TEST_QUERY_ID: json.loads(_VALUES)}

  @parameterized.named_parameters(
      dict(
          testcase_name="one tipping point",
          query=_TEST_QUERY_0,
          expected_row_count=6,
          expected_plan_tipping_point_increase=14,
          expected_plan_tipping_point_decrease=None,
          expected_default_hints="/*+  SeqScan(foo) SeqScan(baz) SeqScan(bar) HashJoin(baz bar) HashJoin(foo baz bar) Leading((foo (baz bar))) */",
          expected_plan_tipping_point_increase_hints="/*+  SeqScan(baz) SeqScan(foo) SeqScan(bar) HashJoin(foo bar) HashJoin(baz foo bar) Leading((baz (foo bar))) */",
          expected_plan_tipping_point_decrease_hints=None),
      dict(
          testcase_name="two tipping points",
          query=_TEST_QUERY_1,
          expected_row_count=6,
          expected_plan_tipping_point_increase=64,
          expected_plan_tipping_point_decrease=5,
          expected_default_hints="/*+  SeqScan(foo) SeqScan(baz) SeqScan(bar) HashJoin(baz bar) HashJoin(foo baz bar) Leading((foo (baz bar))) */",
          expected_plan_tipping_point_increase_hints="/*+  SeqScan(baz) SeqScan(bar) SeqScan(foo) NestLoop(bar foo) HashJoin(baz bar foo) Leading((baz (bar foo))) */",
          expected_plan_tipping_point_decrease_hints="/*+  SeqScan(foo) SeqScan(baz) SeqScan(bar) HashJoin(baz bar) NestLoop(foo baz bar) Leading((foo (baz bar))) */"
      ))
  def test_calculate_plan_changing_cardinality_estimates(
      self, query: str, expected_row_count: int,
      expected_plan_tipping_point_increase: Optional[int],
      expected_plan_tipping_point_decrease: Optional[int],
      expected_default_hints: str,
      expected_plan_tipping_point_increase_hints: Optional[str],
      expected_plan_tipping_point_decrease_hints: Optional[str]):
    templates = {self._query_id: json.loads(query)}

    results = pg_execute_explain_tools.calculate_plan_changing_cardinality_estimates(
        database_configuration=self._database_configuration,
        query_id=self._query_id,
        templates=templates,
        parameter_values=self._values,
        limit=1,
        multiprocessing_chunksize=1)

    results_per_id = results[next(iter(results))]
    results_per_param = results_per_id[next(iter(results_per_id))]

    self.assertLen(results_per_param, 1)
    results_per_join = results_per_param[2]
    self.assertLen(results_per_join, 4)
    self.assertEqual(results_per_join["row_count"], expected_row_count)
    self.assertEqual(results_per_join["plan_tipping_point_increase"],
                     expected_plan_tipping_point_increase)
    self.assertEqual(results_per_join["plan_tipping_point_decrease"],
                     expected_plan_tipping_point_decrease)

    # Show the plans are different at the tipping points.
    query = templates[self._query_id]["query"]
    tables = "bar baz"
    params = self._values[self._query_id]["params"][0]
    default_hints, _ = pg_plan_hint_extractor.get_single_query_hints_with_plan(
        query_manager=self._query_manager,
        query=pg_execute_explain_tools._get_row_hinted_query(
            query=query, tables=tables, row_count=expected_row_count),
        params=params)
    self.assertEqual(default_hints, expected_default_hints)

    if expected_plan_tipping_point_increase is not None:
      plan_tipping_point_increase_hints, _ = pg_plan_hint_extractor.get_single_query_hints_with_plan(
          query_manager=self._query_manager,
          query=pg_execute_explain_tools._get_row_hinted_query(
              query=query,
              tables=tables,
              row_count=expected_plan_tipping_point_increase),
          params=params)
      self.assertEqual(plan_tipping_point_increase_hints,
                       expected_plan_tipping_point_increase_hints)
      self.assertNotEqual(default_hints, plan_tipping_point_increase_hints)

    if expected_plan_tipping_point_decrease is not None:
      plan_tipping_point_decrease_hints, _ = pg_plan_hint_extractor.get_single_query_hints_with_plan(
          query_manager=self._query_manager,
          query=pg_execute_explain_tools._get_row_hinted_query(
              query=query,
              tables=tables,
              row_count=expected_plan_tipping_point_decrease),
          params=params)
      self.assertEqual(plan_tipping_point_decrease_hints,
                       expected_plan_tipping_point_decrease_hints)
      self.assertNotEqual(default_hints, plan_tipping_point_decrease_hints)

    if expected_plan_tipping_point_increase is not None and expected_plan_tipping_point_decrease is not None:
      # This is true for the test cases. There may exist queries for which this
      # property does not universally hold true.
      self.assertNotEqual(plan_tipping_point_increase_hints,
                          plan_tipping_point_decrease_hints)

  def test_calculate_plan_changing_cardinality_estimates_no_joins(self):
    templates = {self._query_id: json.loads(_TEST_QUERY_NO_JOINS)}

    results = pg_execute_explain_tools.calculate_plan_changing_cardinality_estimates(
        database_configuration=self._database_configuration,
        query_id=self._query_id,
        templates=templates,
        parameter_values=self._values,
        limit=1,
        multiprocessing_chunksize=1)

    results_per_id = results[next(iter(results))]
    results_per_param = results_per_id[next(iter(results_per_id))]

    self.assertEmpty(results_per_param)

  @parameterized.named_parameters(
      dict(testcase_name="no limit", limit=None),
      dict(testcase_name="small limit", limit=1),
      dict(testcase_name="limit includes all", limit=3),
      dict(testcase_name="limit is too large", limit=4),
  )
  def test_calculate_plan_changing_cardinality_estimates_limit(
      self, limit: int):
    templates = {test_util.TEST_QUERY_ID: json.loads(_TEST_QUERY_0)}
    results = pg_execute_explain_tools.calculate_plan_changing_cardinality_estimates(
        database_configuration=self._database_configuration,
        query_id=self._query_id,
        templates=templates,
        parameter_values=self._values,
        limit=limit,
        multiprocessing_chunksize=1)

    param_keys = [
        "####".join([str(element)
                     for element in param])
        for param in self._values[self._query_id]["params"][:limit]
    ]

    self.assertLen(results, 1)
    self.assertEqual(set(results[self._query_id].keys()), set(param_keys))

  @parameterized.named_parameters(
      dict(
          testcase_name="1 join",
          query=_TEST_QUERY_JOINS_1,
          expected_join_count_keys=set([])),
      dict(
          testcase_name="2 joins",
          query=_TEST_QUERY_JOINS_2,
          expected_join_count_keys=set([2])),
      dict(
          testcase_name="3 joins",
          query=_TEST_QUERY_JOINS_3,
          expected_join_count_keys=set([2, 3])),
  )
  def test_calculate_plan_changing_cardinality_estimates_join_counts(
      self, query: str, expected_join_count_keys: Set[int]):
    templates = {test_util.TEST_QUERY_ID: json.loads(query)}
    results = pg_execute_explain_tools.calculate_plan_changing_cardinality_estimates(
        database_configuration=self._database_configuration,
        query_id=self._query_id,
        templates=templates,
        parameter_values=self._values,
        limit=1,
        multiprocessing_chunksize=1)

    results_per_id = results[next(iter(results))]
    results_per_param = results_per_id[next(iter(results_per_id))]

    self.assertEqual(set(results_per_param.keys()), expected_join_count_keys)

  @parameterized.named_parameters(
      dict(testcase_name="no limit", limit=None),
      dict(testcase_name="small limit", limit=1),
      dict(testcase_name="limit includes all", limit=3),
      dict(testcase_name="limit is too large", limit=4),
  )
  def test_collect_explain_plan_total_costs(self, limit: int):
    templates = {test_util.TEST_QUERY_ID: json.loads(_TEST_QUERY_0)}
    results = pg_execute_explain_tools.collect_explain_plan_info(
        database_configuration=self._database_configuration,
        query_id=self._query_id,
        templates=templates,
        parameter_values=self._values,
        plan_hints=self._hints,
        extract_function=pg_execute_explain_tools.ExplainExtractionFunction
        .TOTAL_COSTS,
        limit=limit,
        multiprocessing_chunksize=1)

    param_keys = [
        "####".join([str(element)
                     for element in param])
        for param in self._values[self._query_id]["params"][:limit]
    ]

    self.assertLen(results, 1)
    self.assertEqual(set(results[self._query_id].keys()), set(param_keys))

    # Verify there are as many costs as plans and that the costs are distinct
    # for each plan on a per-parameter basis.
    for param_results in results[self._query_id].values():
      self.assertLen(param_results["results"], len(self._hints[self._query_id]))
      costs = set()
      for entry in param_results["results"]:
        self.assertLen(entry, 1)
        costs.add(entry[0]["total_cost"])

      self.assertLen(costs, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name="query_0",
          query=_TEST_QUERY_0,
          expected=["foo", "bar", "baz"]),
      dict(
          testcase_name="query_no_joins",
          query=_TEST_QUERY_NO_JOINS,
          expected=["bar"]),
      dict(
          testcase_name="query",
          query=_TEST_QUERY_JOINS_1,
          expected=["foo_a", "foo_b"]),
  )
  def test_collect_explain_plan_row_counts(self, query: str,
                                           expected: List[str]):
    templates = {test_util.TEST_QUERY_ID: json.loads(query)}
    results = pg_execute_explain_tools.collect_explain_plan_info(
        database_configuration=self._database_configuration,
        query_id=self._query_id,
        templates=templates,
        parameter_values=self._values,
        plan_hints=self._hints,
        extract_function=pg_execute_explain_tools.ExplainExtractionFunction
        .ESTIMATED_CARDINALITIES,
        multiprocessing_chunksize=1)

    param_keys = [
        "####".join([str(element)
                     for element in param])
        for param in self._values[self._query_id]["params"]
    ]

    self.assertLen(results, 1)
    self.assertEqual(set(results[self._query_id].keys()), set(param_keys))

    # Verify there are as many results as plans.
    for param_results in results[self._query_id].values():
      self.assertLen(param_results["results"], len(self._hints[self._query_id]))

      for entry in param_results["results"]:
        self.assertLen(entry, 1)
        self.assertEqual(
            set(entry[0]["estimated_cardinalities"].keys()), set(expected))


if __name__ == "__main__":
  absltest.main()

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

"""Tests for pg_perturb_plan_cardinalities.py."""

import itertools
import json
from typing import Any, List

from kepler.training_data_collection_pipeline import pg_perturb_plan_cardinalities
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util

from absl.testing import absltest

# Typing aliases.
JSON = Any

_TEST_QUERY_0 = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo JOIN bar on x = b WHERE \\n bar.a < @param0 \\n and bar.c = '@param1' \\n order by y;"
}
"""

_TEST_QUERY_1 = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo JOIN bar on x = b \\n JOIN baz on x = j WHERE \\n bar.a < @param0 \\n and bar.c = '@param1' \\n order by y;"
}
"""

_HINTS_QUERY_0 = """
[
  {"hints": "/*+  SeqScan(foo) SeqScan(bar) MergeJoin(foo bar) Leading((foo bar)) */", "source": "default"},
  {"hints": "/*+  SeqScan(foo) SeqScan(bar) HashJoin(foo bar) Leading((foo bar)) */", "source": "default"}
]
"""

_HINTS_QUERY_1 = """
[
  {"hints": "/*+  SeqScan(foo) SeqScan(bar) SeqScan(baz) MergeJoin(foo bar) MergeJoin(foo bar baz) Leading(((foo bar) baz)) */", "source": "default"},
  {"hints": "/*+  SeqScan(bar) SeqScan(foo) SeqScan(baz) HashJoin(foo baz) HashJoin(bar foo baz) Leading((bar (foo baz))) */", "source": "default"}
]
"""

_VALUES = """
[
  {"params": [0, "alpha"], "plan_index": "0"},
  {"params": [1, "bravo"], "plan_index": "1"},
  {"params": [1, "charlie"], "plan_index": "0"}
]
"""


def _extract_plan_join_cardinalties_helper(
    plan: JSON, join_node_type: str, join_cardinalites: List[int]) -> None:
  if "Plans" in plan:
    for child_plan in plan["Plans"]:
      _extract_plan_join_cardinalties_helper(child_plan, join_node_type,
                                             join_cardinalites)

  if plan["Node Type"] == join_node_type:
    join_cardinalites.append(plan["Plan Rows"])


def _extract_plan_join_cardinalties(plan: JSON,
                                    join_node_type: str) -> List[int]:
  """Extracts the estimated join cardinalities from the plan.

  Args:
    plan: The EXPLAIN plan.
    join_node_type: The node type of the join for which cardinality estimates
      are being extracted. These tests use the same node type for all joins in a
      single plan.

  Returns:
    A list containing the join estimates in the order they are encountered by
    a post-order tree walk.
  """
  join_cardinalites = []
  _extract_plan_join_cardinalties_helper(plan["Plan"], join_node_type,
                                         join_cardinalites)
  return join_cardinalites


class PgExecuteTrainingDataQueriesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._query_manager = query_utils.QueryManager(
        query_utils.DatabaseConfiguration(
            dbname=self._test_database.dbname,
            user=test_util.USER,
            password=test_util.PASSWORD))
    test_util.populate_database(self._query_manager)

    self.query_id = test_util.TEST_QUERY_ID
    self.values = {test_util.TEST_QUERY_ID: json.loads(_VALUES)}
    self.params = [
        value["params"] for value in self.values[test_util.TEST_QUERY_ID]
    ]

  def test_multiplicatively_perturb_plan_cardinalities_simple_plan(self):
    """Verifies cardinality hints are applied to a single join output.

    Additionally verifies that cardinality hints are perturbed for each set of
    plan hints and limit number of parameter values.
    """
    templates = {test_util.TEST_QUERY_ID: json.loads(_TEST_QUERY_0)}
    hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS_QUERY_0)}

    cardinality_multipliers = [.01, .4, 10]
    # The cardinality estimate for the join without row count adjustments is 11
    # for both sets of hints. The elements of expected_join_cardinalities are
    # produced by multiplying the elements of cardinality_multipliers by 11 and
    # applying adjustments as described below.
    #
    # Test cases:
    # 1) Values below 1 are forced up to 1.
    # 2) Decimal values are rounded.
    # 3) A multiplication to a higher integer scales the estimate directly.
    expected_join_cardinalities = [[1], [4], [110]]

    parameter_limit = 2
    plans = pg_perturb_plan_cardinalities.multiplicatively_perturb_plan_cardinalities(
        query_manager=self._query_manager,
        query_id=self.query_id,
        templates=templates,
        parameter_values=self.values,
        plan_hints=hints,
        cardinality_multipliers=cardinality_multipliers,
        limit=parameter_limit,
        multiprocessing_chunksize=1)

    self.assertLen(plans, 1)
    self.assertIn(self.query_id, plans)

    param_keys = [
        "####".join([str(element)
                     for element in param])
        for param in self.params[:parameter_limit]
    ]
    join_node_types = ["Merge Join", "Hash Join"]
    self.assertLen(plans[self.query_id], parameter_limit)

    costs = []
    for param_key in param_keys:
      self.assertIn(param_key, plans[self.query_id])
      results = plans[self.query_id][param_key]["results"]
      # There are results for 2 hints per param_key.
      self.assertLen(results, 2)
      for result, join_node_type in zip(results, join_node_types):
        # Verify data was generated for a single iteration.
        self.assertLen(result, 1)
        plan_perturbations = result[0]["explain_output_across_cardinality"]
        # Each hint has len(cardinality_multipliers)^num_joins explain plans. In
        # this case, that is 3^1 = 3.
        self.assertLen(plan_perturbations, 3)
        actual_join_cardinalities = []
        for plan in plan_perturbations:
          join_cardinalities = _extract_plan_join_cardinalties(
              plan, join_node_type)
          actual_join_cardinalities.append(join_cardinalities)
          costs.append(plan["Plan"]["Total Cost"])
        self.assertEqual(actual_join_cardinalities, expected_join_cardinalities)

    # For a given parameter, we should have a distinct cost per plan because of
    # the differences in cardinality estimates, plan types, and the sort on a
    # non-join column after the final join output. The 2 parameter values
    # produce the same cost profiles. Checking the exact costs verifies the
    # determinisim in the database configuration and the execution ordering.
    expected_costs = [190.8, 190.84, 194.79, 62.01, 62.05, 66.0] * 2
    self.assertEqual(costs, expected_costs)

  def test_multiplicatively_perturb_plan_cardinalities_multiple_joins(self):
    """Verifies cardinality hints are applied when there are multiple joins."""
    templates = {test_util.TEST_QUERY_ID: json.loads(_TEST_QUERY_1)}
    hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS_QUERY_1)}
    cardinality_multipliers = [.01, .4, 10]

    # The expected join cardinalities for the 2-table and 3-table join
    # respectively for each hint. The 2 hints are differentiated by the join
    # type. The expected output will be the cross-product of the 2-table and
    # 3-table join cardinalities.
    expected_join_cardinality_summary = {
        "Merge Join": [[1, 4, 110], [1, 26, 660]],
        "Hash Join": [[135, 5424, 135600], [1, 26, 660]]
    }

    parameter_limit = 1
    plans = pg_perturb_plan_cardinalities.multiplicatively_perturb_plan_cardinalities(
        query_manager=self._query_manager,
        query_id=self.query_id,
        templates=templates,
        parameter_values=self.values,
        plan_hints=hints,
        cardinality_multipliers=cardinality_multipliers,
        limit=parameter_limit,
        multiprocessing_chunksize=1)

    self.assertLen(plans, 1)
    self.assertIn(self.query_id, plans)

    costs = []
    param_key = "####".join([str(element) for element in self.params[0]])
    self.assertLen(plans[self.query_id], parameter_limit)
    self.assertIn(param_key, plans[self.query_id])
    results = plans[self.query_id][param_key]["results"]
    # There are results for 2 hints.
    self.assertLen(results, 2)

    join_node_types = ["Merge Join", "Hash Join"]
    for result, join_node_type in zip(results, join_node_types):
      # Verify data was generated for a single iteration.
      self.assertLen(result, 1)
      plan_perturbations = result[0]["explain_output_across_cardinality"]
      # Each hint has len(cardinality_multipliers)^num_joins explain plans. In
      # this case, that is 3^2 = 9.
      self.assertLen(plan_perturbations, 9)
      expected_join_cardinalities = list(
          itertools.product(
              expected_join_cardinality_summary[join_node_type][0],
              expected_join_cardinality_summary[join_node_type][1]))
      actual_join_cardinalities = []
      for plan in plan_perturbations:
        join_cardinalities = _extract_plan_join_cardinalties(
            plan, join_node_type)
        actual_join_cardinalities.append(tuple(join_cardinalities))
        costs.append(plan["Plan"]["Total Cost"])
      self.assertEqual(actual_join_cardinalities,
                       list(expected_join_cardinalities))

    # Each plan should have a distinct cost because of the differences in
    # cardinality estimates, plan types, and the sort on a non-join column after
    # the final join output. Checking the exact costs verifies the determinisim
    # in the database configuration and the execution ordering.
    expected_costs = [
        280.24, 280.9, 312.78, 280.42, 281.08, 312.97, 287.05, 287.71, 319.59,
        572.38, 573.04, 604.93, 639.41, 640.08, 671.96, 2289.4, 2290.06, 2321.94
    ]
    self.assertEqual(costs, expected_costs)

  def test_multiplicatively_perturb_plan_cardinalities_keys_to_remove(self):
    """Verifies EXPLAIN plans are stripped of keys_to_remove to reduce size."""
    templates = {test_util.TEST_QUERY_ID: json.loads(_TEST_QUERY_0)}
    hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS_QUERY_0)}

    cardinality_multipliers = [.01, 10]

    parameter_limit = 1
    plans = pg_perturb_plan_cardinalities.multiplicatively_perturb_plan_cardinalities(
        query_manager=self._query_manager,
        query_id=self.query_id,
        templates=templates,
        parameter_values=self.values,
        plan_hints=hints,
        cardinality_multipliers=cardinality_multipliers,
        limit=parameter_limit,
        keys_to_remove=["Node Type"],
        multiprocessing_chunksize=1)

    self.assertLen(plans, 1)
    self.assertIn(self.query_id, plans)

    param_key = "####".join([str(element) for element in self.params[0]])
    self.assertLen(plans[self.query_id], parameter_limit)
    self.assertIn(param_key, plans[self.query_id])
    results = plans[self.query_id][param_key]["results"]
    # There are results for 2 hints.
    self.assertLen(results, 2)

    for result in results:
      # Verify data was generated for a single iteration.
      self.assertLen(result, 1)
      plan_perturbations = result[0]["explain_output_across_cardinality"]
      for plan in plan_perturbations:
        self.assertNotIn("Node Type", plan)


if __name__ == "__main__":
  absltest.main()

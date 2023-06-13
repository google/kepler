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

"""Tests for evaluation module."""

from typing import Any, Dict, List, Optional, Set

import numpy as np

from kepler.data_management import database_simulator
from kepler.data_management import test_util
from kepler.data_management import workload
from kepler.model_trainer import evaluation
from absl.testing import absltest
from absl.testing import parameterized

# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = "####"

_DELTA = 1e-4

_DEFAULT = "default"
_RESULTS = "results"
_DURATION_MS = "duration_ms"
_TIMED_OUT = "timed_out"
_SKIPPED = "skipped"


def _get_param_as_string(params: List[str]) -> str:
  return _NAME_DELIMITER.join(params)


def _get_min_result(result_list: List[Any]) -> Optional[float]:
  values = []
  for result in result_list:
    if _DURATION_MS in result:
      values.append(result[_DURATION_MS])
    if _TIMED_OUT in result:
      return result[_TIMED_OUT]
    if _SKIPPED in result:
      return None
  return np.min(values)


def _compute_test_util_default_latencies(
    query_workload: workload.Workload) -> List[float]:
  data_by_param = test_util.QUERY_EXECUTION_DATA[test_util.TEST_QUERY_ID]
  latencies = []
  for query_instance in query_workload.query_log:
    param_entry = data_by_param[_get_param_as_string(query_instance.parameters)]
    latency = _get_min_result(param_entry[_RESULTS][param_entry[_DEFAULT]])
    assert latency is not None
    latencies.append(latency)

  return latencies


def _compute_test_util_optimal_latencies(
    query_workload: workload.Workload) -> List[float]:
  data_by_param = test_util.QUERY_EXECUTION_DATA[test_util.TEST_QUERY_ID]
  latencies = []
  for query_instance in query_workload.query_log:
    param_entry = data_by_param[_get_param_as_string(query_instance.parameters)]
    param_latencies = [
        _get_min_result(result) for result in param_entry[_RESULTS]
    ]
    latencies.append(
        np.min([
            param_latency for param_latency in param_latencies
            if param_latency is not None
        ]))

  return latencies


def _compute_test_util_candidate_latencies(
    query_workload: workload.Workload,
    plan_selections: List[int]) -> List[float]:
  assert len(query_workload.query_log) == len(plan_selections)
  data_by_param = test_util.QUERY_EXECUTION_DATA[test_util.TEST_QUERY_ID]
  latencies = []
  for query_instance, plan_selection in zip(query_workload.query_log,
                                            plan_selections):
    param_entry = data_by_param[_get_param_as_string(query_instance.parameters)]
    plan_index = plan_selection if plan_selection is not None else param_entry[
        _DEFAULT]
    latency = _get_min_result(param_entry[_RESULTS][plan_index])
    assert latency is not None
    latencies.append(latency)

  return latencies


class Evaluation(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA)
    self._database = database_simulator.DatabaseSimulator(
        query_execution_data=test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)
    self._client = database_simulator.DatabaseClient(self._database)

    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)
    self._workload_all = workload_generator.all()
    self._workload_empty = workload.Workload(
        query_id=test_util.TEST_QUERY_ID, query_log=[])

  def test_get_default_latencies(self):
    default_latencies = evaluation.get_default_latencies(
        database=self._database, query_workload=self._workload_all)
    expected = _compute_test_util_default_latencies(self._workload_all)
    self.assertEqual(default_latencies, expected)

    default_latencies_empty = evaluation.get_default_latencies(
        database=self._database, query_workload=self._workload_empty)
    self.assertEmpty(default_latencies_empty)

  def test_get_optimal_latencies(self):
    optimal_latencies = evaluation.get_optimal_latencies(
        client=self._client,
        query_workload=self._workload_all,
        kepler_plan_discoverer=self._plans)
    expected = _compute_test_util_optimal_latencies(self._workload_all)
    self.assertEqual(optimal_latencies, expected)

    optimal_latencies_empty = evaluation.get_optimal_latencies(
        client=self._client,
        query_workload=self._workload_empty,
        kepler_plan_discoverer=self._plans)
    self.assertEmpty(optimal_latencies_empty)

  def test_get_candidate_latencies(self):
    plan_selections = [3, 1, None, 2, None]
    candidate_latencies = evaluation.get_candidate_latencies(
        database=self._database,
        query_workload=self._workload_all,
        plan_selections=np.array(plan_selections))
    expected = _compute_test_util_candidate_latencies(self._workload_all,
                                                      plan_selections)
    self.assertEqual(candidate_latencies, expected)

    candidate_latencies_empty = evaluation.get_candidate_latencies(
        database=self._database,
        query_workload=self._workload_empty,
        plan_selections=np.array([]))
    self.assertEmpty(candidate_latencies_empty)

  def test_get_candidate_latencies_illegal_call(self):
    self.assertRaisesRegex(
        ValueError,
        "The query_workload query log must contain the same number",
        evaluation.get_candidate_latencies,
        self._database,
        self._workload_all,
        plan_selections=[0])

  def test_evaluate_near_optimal_and_improvement_threshold(self):
    candidate_latencies = evaluation.get_candidate_latencies(
        database=self._database,
        query_workload=self._workload_all,
        plan_selections=np.array([0, 0, 0, 0, 0]))
    default_latencies = evaluation.get_default_latencies(
        database=self._database, query_workload=self._workload_all)
    optimal_latencies = evaluation.get_optimal_latencies(
        client=self._client,
        query_workload=self._workload_all,
        kepler_plan_discoverer=self._plans)

    # Plan 0 is optimal for first two params, 2x worse than default for 3rd
    # param, is default for 4th param, times out for 5th param (5x worse).
    results = evaluation.evaluate(
        candidate_latencies=candidate_latencies,
        default_latencies=default_latencies,
        optimal_latencies=optimal_latencies)
    self.assertEmpty(results[evaluation.IMPROVEMENTS_RELATIVE])
    self.assertEqual(results[evaluation.REGRESSIONS_RELATIVE], [2, 50])
    self.assertEmpty(results[evaluation.IMPROVEMENTS_ABSOLUTE])
    self.assertEqual(results[evaluation.REGRESSIONS_ABSOLUTE], [111, 49])
    self.assertEqual(results[evaluation.NUM_EQUIVALENT], 3)
    self.assertEqual(results[evaluation.NUM_NEAR_OPTIMAL], 2)

    results = evaluation.evaluate(
        candidate_latencies=candidate_latencies,
        default_latencies=default_latencies,
        optimal_latencies=optimal_latencies,
        improvement_threshold=2.01)
    self.assertEmpty(results[evaluation.IMPROVEMENTS_RELATIVE])
    self.assertEqual(results[evaluation.REGRESSIONS_RELATIVE], [50])
    self.assertEmpty(results[evaluation.IMPROVEMENTS_ABSOLUTE])
    self.assertEqual(results[evaluation.REGRESSIONS_ABSOLUTE], [49])
    self.assertEqual(results[evaluation.NUM_EQUIVALENT], 4)
    self.assertEqual(results[evaluation.NUM_NEAR_OPTIMAL], 2)

    results = evaluation.evaluate(
        candidate_latencies=candidate_latencies,
        default_latencies=default_latencies,
        optimal_latencies=optimal_latencies,
        near_optimal_threshold=.4)
    self.assertEmpty(results[evaluation.IMPROVEMENTS_RELATIVE])
    self.assertEqual(results[evaluation.REGRESSIONS_RELATIVE], [2, 50])
    self.assertEmpty(results[evaluation.IMPROVEMENTS_ABSOLUTE])
    self.assertEqual(results[evaluation.REGRESSIONS_ABSOLUTE], [111, 49])
    self.assertEqual(results[evaluation.NUM_EQUIVALENT], 3)
    self.assertEqual(results[evaluation.NUM_NEAR_OPTIMAL], 3)

    # Plan 0 is optimal for first two params, 2x worse than default for 3rd
    # param, times out for 5th param (5x worse). Plan 2 is optimal for the 4th
    # and 3x better than default.
    candidate_latencies_beats_default = evaluation.get_candidate_latencies(
        database=self._database,
        query_workload=self._workload_all,
        plan_selections=np.array([0, 0, 0, 2, 0]))
    results = evaluation.evaluate(
        candidate_latencies=candidate_latencies_beats_default,
        default_latencies=default_latencies,
        optimal_latencies=optimal_latencies)
    self.assertEqual(results[evaluation.IMPROVEMENTS_RELATIVE], [3])
    self.assertEqual(results[evaluation.REGRESSIONS_RELATIVE], [2, 50])
    self.assertEqual(results[evaluation.IMPROVEMENTS_ABSOLUTE], [2222])
    self.assertEqual(results[evaluation.REGRESSIONS_ABSOLUTE], [111, 49])
    self.assertEqual(results[evaluation.NUM_EQUIVALENT], 2)
    self.assertEqual(results[evaluation.NUM_NEAR_OPTIMAL], 3)

    results = evaluation.evaluate(
        candidate_latencies=candidate_latencies_beats_default,
        default_latencies=default_latencies,
        optimal_latencies=optimal_latencies,
        improvement_threshold=4)
    self.assertEmpty(results[evaluation.IMPROVEMENTS_RELATIVE])
    self.assertEqual(results[evaluation.REGRESSIONS_RELATIVE], [50])
    self.assertEmpty(results[evaluation.IMPROVEMENTS_ABSOLUTE])
    self.assertEqual(results[evaluation.REGRESSIONS_ABSOLUTE], [49])
    self.assertEqual(results[evaluation.NUM_EQUIVALENT], 4)
    self.assertEqual(results[evaluation.NUM_NEAR_OPTIMAL], 3)

  def _check_distinctness(self, percentiles: List[float],
                          results: Dict[str, float]) -> None:
    """Verify each percentile value is distinct."""

    self.assertLen(results, len(percentiles))
    values = set([
        results[evaluation._format_percentile_key(percentile)]
        for percentile in percentiles
    ])
    self.assertLen(values, len(percentiles))

  def test_evaluate_general_stats(self):
    candidate_latencies = evaluation.get_candidate_latencies(
        database=self._database,
        query_workload=self._workload_all,
        plan_selections=np.array([0, 0, 0, 2, 0]))
    default_latencies = evaluation.get_default_latencies(
        database=self._database, query_workload=self._workload_all)
    optimal_latencies = evaluation.get_optimal_latencies(
        client=self._client,
        query_workload=self._workload_all,
        kepler_plan_discoverer=self._plans)

    percentiles = [50, 90, 92, 100]
    results = evaluation.evaluate(
        candidate_latencies=candidate_latencies,
        default_latencies=default_latencies,
        optimal_latencies=optimal_latencies,
        percentiles=percentiles)

    self.assertAlmostEqual(
        results[evaluation.MEAN_CANDIDATE_LATENCY],
        np.mean(candidate_latencies),
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.MEAN_DEFAULT_LATENCY],
        np.mean(default_latencies),
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.MEAN_OPTIMAL_LATENCY],
        np.mean(optimal_latencies),
        delta=_DELTA)

    self.assertAlmostEqual(
        results[evaluation.CANDIDATE_SUBOPTIMALITY],
        np.mean(candidate_latencies) / np.mean(optimal_latencies),
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.DEFAULT_SUBOPTIMALITY],
        np.mean(default_latencies) / np.mean(optimal_latencies),
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.CANDIDATE_SPEEDUP],
        np.mean(default_latencies) / np.mean(candidate_latencies),
        delta=_DELTA)

    # Spot check percentile exact values for p100.
    self.assertAlmostEqual(
        results[evaluation.CANDIDATE_TAIL_SUBOPTIMALITY][
            evaluation._format_percentile_key(100)],
        np.max(candidate_latencies) / np.max(optimal_latencies),
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.DEFAULT_TAIL_SUBOPTIMALITY][
            evaluation._format_percentile_key(100)],
        np.max(default_latencies) / np.max(optimal_latencies),
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.CANDIDATE_TAIL_SPEEDUP][
            evaluation._format_percentile_key(100)],
        np.max(default_latencies) / np.max(candidate_latencies),
        delta=_DELTA)

    # Ensure all percentile values are distinct.
    self._check_distinctness(
        percentiles=percentiles,
        results=results[evaluation.CANDIDATE_TAIL_SUBOPTIMALITY])
    self._check_distinctness(
        percentiles=percentiles,
        results=results[evaluation.DEFAULT_TAIL_SUBOPTIMALITY])
    self._check_distinctness(
        percentiles=percentiles,
        results=results[evaluation.CANDIDATE_TAIL_SPEEDUP])

    # Spot check tail cdf speedups for p50.
    self.assertAlmostEqual(
        results[evaluation.CANDIDATE_TAIL_CDF_SPEEDUP][
            evaluation._format_percentile_key(50)],
        3455 / 1344,
        delta=_DELTA)
    self.assertAlmostEqual(
        results[evaluation.OPTIMAL_TAIL_CDF_SPEEDUP][
            evaluation._format_percentile_key(50)],
        3455 / 1233,
        delta=_DELTA)

    # Check that candidate tail cdf speedups are less than optimal.
    for percentile_key in results[evaluation.OPTIMAL_TAIL_CDF_SPEEDUP]:
      self.assertLessEqual(
          results[evaluation.CANDIDATE_TAIL_CDF_SPEEDUP][percentile_key],
          results[evaluation.OPTIMAL_TAIL_CDF_SPEEDUP][percentile_key])

  def test_evaluate_illegal_calls(self):
    self.assertRaisesRegex(
        ValueError,
        "All of candidate_latencies, default_latencies, and optimal_latencies",
        evaluation.evaluate, [], [1], [])

    self.assertRaisesRegex(
        ValueError,
        "All of candidate_latencies, default_latencies, and optimal_latencies",
        evaluation.evaluate, [1], [1], [])

    self.assertRaisesRegex(
        ValueError,
        "All of candidate_latencies, default_latencies, and optimal_latencies",
        evaluation.evaluate, [], [2, 1], [2, 1])

    self.assertRaisesRegex(ValueError, "No latency values to evaluate",
                           evaluation.evaluate, [], [], [])


COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXECUTION_DATA = {
    test_util.TEST_QUERY_ID: {
        "first": {
            "default":
                0,
            "results": [[{
                "duration_ms": 1
            }], [{
                "duration_ms": 2
            }], [{
                "duration_ms": 3
            }], [{
                "duration_ms": 4
            }], [{
                "duration_ms": 5
            }]]
        },
        "second": {
            "default":
                0,
            "results": [[{
                "duration_ms": 3
            }], [{
                "duration_ms": 4
            }], [{
                "duration_ms": 2
            }], [{
                "duration_ms": 1
            }], [{
                "duration_ms": 5
            }]]
        },
        "third": {
            "default":
                3,
            "results": [[{
                "duration_ms": 1
            }], [{
                "duration_ms": 2
            }], [{
                "duration_ms": 3
            }], [{
                "duration_ms": 4
            }], [{
                "duration_ms": 5
            }]]
        },
        "fourth": {
            "default":
                3,
            "results": [[{
                "duration_ms": 4
            }], [{
                "duration_ms": 5
            }], [{
                "duration_ms": 1
            }], [{
                "duration_ms": 2
            }], [{
                "duration_ms": 3
            }]]
        },
        "fifth": {
            "default":
                2,
            "results": [[{
                "duration_ms": 3
            }], [{
                "duration_ms": 1
            }], [{
                "duration_ms": 4
            }], [{
                "duration_ms": 2
            }], [{
                "duration_ms": 5
            }]]
        },
    }
}

COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXECUTION_METADATA = {
    test_util.TEST_QUERY_ID: {
        "plan_cover": [0, 1, 2, 3, 4]
    }
}

COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXPLAIN_DATA = {
    test_util.TEST_QUERY_ID: {
        "first": {
            "results": [[{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 999
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }]]
        },
        "second": {
            "results": [[{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1
            }]]
        },
        "third": {
            "results": [[{
                "total_cost": 100
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }]]
        },
        "fourth": {
            "results": [[{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 10
            }], [{
                "total_cost": 1000
            }]]
        },
        "fifth": {
            "results": [[{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1000
            }], [{
                "total_cost": 1
            }]]
        },
    }
}


class CostBasedCandidatePlanSetEvaluatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._plans = workload.KeplerPlanDiscoverer(
        query_execution_data=COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXPLAIN_DATA
    )
    database = database_simulator.DatabaseSimulator(
        query_execution_data=COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXECUTION_DATA,
        query_execution_metadata=COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXPLAIN_DATA
    )
    self._client = database_simulator.DatabaseClient(database)
    self._workload = workload.WorkloadGenerator(
        COST_BASED_CANDIDATE_PLAN_SET_EVALUATOR_QUERY_EXECUTION_DATA,
        test_util.TEST_SEED).all()

    self._evaluator = evaluation.CostBasedCandidatePlanSetEvaluator(
        client=self._client,
        query_workload=self._workload,
        kepler_plan_discoverer=self._plans,
        use_default_plans_only=True)

  def test_num_plans(self):
    self.assertEqual(self._evaluator.num_plans, 3)

  # The cost of 1 for plan 2 for "second" makes it the initial cheapest plan,
  # being orders of magnitude lower than the best costs for plans 0 and 3. The
  # cost of 10 for "fourth" allows plan 3 to beat plan 1, whose best cost
  # contribution is a 100 for "third". Plan 5 is twice the best but it's not a
  # default plan and hence cannot be in the plan set.
  @parameterized.named_parameters(
      dict(testcase_name="size 0", plan_set_size=0, plan_set=set()),
      dict(testcase_name="size 1", plan_set_size=1, plan_set=set([2])),
      dict(testcase_name="size 2", plan_set_size=2, plan_set=set([2, 3])),
      dict(testcase_name="size 3", plan_set_size=3, plan_set=set([0, 2, 3])))
  def test_populate_cache_greedy(self, plan_set_size: int, plan_set: Set[int]):
    self.assertEqual(
        self._evaluator.populate_cache_greedy(plan_set_size), plan_set)

  # The cost of 1 (twice) for plan 4 makes it the best. Then plan 3 offers an
  # improvement. Although plan 2 has a cost of 1, its low cost parameter
  # overlaps with plan 4 and does not provide any additional value. After plan
  # 3, we get plan 0 as before. Plan 2 is finally able to provide marginal value
  # above plan 1.
  @parameterized.named_parameters(
      dict(testcase_name="size 0", plan_set_size=0, plan_set=set()),
      dict(testcase_name="size 1", plan_set_size=1, plan_set=set([4])),
      dict(testcase_name="size 2", plan_set_size=2, plan_set=set([3, 4])),
      dict(testcase_name="size 3", plan_set_size=3, plan_set=set([0, 3, 4])),
      dict(testcase_name="size 4", plan_set_size=4, plan_set=set([0, 2, 3, 4])),
      dict(
          testcase_name="size 5",
          plan_set_size=5,
          plan_set=set([0, 1, 2, 3, 4])))
  def test_populate_cache_greedy_all_plans(self, plan_set_size: int,
                                           plan_set: Set[int]):
    evaluator = evaluation.CostBasedCandidatePlanSetEvaluator(
        client=self._client,
        query_workload=self._workload,
        kepler_plan_discoverer=self._plans,
        use_default_plans_only=False)
    self.assertEqual(evaluator.populate_cache_greedy(plan_set_size), plan_set)

  def test_illegal_init(self):
    workload_empty = workload.Workload(
        query_id=test_util.TEST_QUERY_ID, query_log=[])
    self.assertRaisesRegex(
        ValueError,
        "Provide a non-empty query_workload to evaluate.",
        evaluation.CostBasedCandidatePlanSetEvaluator, self._client,
        workload_empty, self._plans, True)

  @parameterized.named_parameters(
      dict(testcase_name="size too low", plan_set_size=-1),
      dict(testcase_name="size too big", plan_set_size=4))
  def test_populate_cache_greedy_illegal_calls(self, plan_set_size: int):
    self.assertRaisesRegex(ValueError, "The plan_set_size must be between ",
                           self._evaluator.populate_cache_greedy, plan_set_size)

  @parameterized.named_parameters(
      dict(
          testcase_name="none",
          plan_set=None,
          expected=[1.0, 1.0, 1.0, 1.0, 2.0]),
      dict(
          testcase_name="size 1",
          plan_set=set([2]),
          expected=[3.0, 2.0, 3.0, 1.0, 4.0]),
      dict(
          testcase_name="size 2",
          plan_set=set([2, 3]),
          expected=[3.0, 1.0, 3.0, 1.0, 2.0]))
  def test_get_latencies(self, plan_set: Optional[Set[int]],
                         expected: List[float]):
    """Verifies getting latencies using the plan sets."""
    self.assertEqual(self._evaluator.get_latencies(plan_set=plan_set), expected)

  def test_get_latencies_all_plans(self):
    """Verifies getting latencies using all plans, not just defaults."""
    evaluator = evaluation.CostBasedCandidatePlanSetEvaluator(
        client=self._client,
        query_workload=self._workload,
        kepler_plan_discoverer=self._plans,
        use_default_plans_only=False)
    expected = [1.0, 1.0, 1.0, 1.0, 1.0]
    self.assertEqual(evaluator.get_latencies(), expected)


if __name__ == "__main__":
  absltest.main()

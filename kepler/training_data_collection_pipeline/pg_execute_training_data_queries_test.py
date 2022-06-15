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

"""Tests for pg_execute_training_data_queries.py."""

import collections
import copy
import dataclasses
import itertools
import json
import math
from typing import Any, List, Optional, Tuple

from kepler.data_management import database_simulator
from kepler.training_data_collection_pipeline import pg_execute_training_data_queries
from kepler.training_data_collection_pipeline import test_util
from absl.testing import absltest
from absl.testing import parameterized

_HINTS = """[{hints_0}, {hints_1}]""".format(
    hints_0=test_util.HINTS_0, hints_1=test_util.HINTS_1)

_VALUES = """
[
  {"params": [0, "alpha"], "plan_index": "0"},
  {"params": [1, "bravo"], "plan_index": "1"},
  {"params": [1, "charlie"], "plan_index": "0"}
]
"""

# Params and hints for finer-grained mocking of execution latency.
# Note: these params must all be unique.
_PARAM_VALUES = [{
    "params": [5, 10, 100],
    "plan_index": "0"
}, {
    "params": [6, 10, 50],
    "plan_index": "0"
}, {
    "params": [50, 25, 70],
    "plan_index": "1"
}, {
    "params": [60, 50, 80],
    "plan_index": "1"
}, {
    "params": [200, 200, 100],
    "plan_index": "2"
}, {
    "params": [3, 10, 10],
    "plan_index": "0"
}, {
    "params": [4, 10, 11],
    "plan_index": "0"
}, {
    "params": [7, 10, 12],
    "plan_index": "0"
}]

_PARAM_HINTS = [{
    "hints": "hint0",
    "source": "default"
}, {
    "hints": "hint1",
    "source": "default"
}, {
    "hints": "hint2",
    "source": "default"
}]

_QUERY_TIMEOUT_MULTIPLIER = 10
_QUERY_TIMEOUT_MIN_MS = 1
_QUERY_TIMEOUT_MAX_MS = 10000

JSON = Any


@dataclasses.dataclass
class QueryExecution:
  query: str
  params: List[Any]


def _execute_hinted_query(
    query: str, params: List[Any],
    timeout_ms: Optional[float]) -> Tuple[Optional[float], Optional[int]]:
  """Executes a hinted query in which the parameter contains the latencies.

  The plan index is extracted from the hint name, of the form hint{plan_index}.

  Args:
    query: Hinted SQL query template string with 0 or more parameters provided
      in the form of @param#, starting with 0.
    params: List of parameter values to substitute into the query. The ith
      value will correspond to the latency for plan i.
    timeout_ms: The statement timeout for this query in ms.

  Returns:
    Tuple:
      1. The execution time of the query in ms or None if the query times
        out.
      2. 1, simulating that this query always returns rows.
  """
  plan_idx = int(query.split(" ")[0][len("hint"):])
  latency = params[plan_idx]

  if timeout_ms and latency >= timeout_ms:
    return None, None

  return latency, 1


def _construct_execution_data(latencies: List[List[List[float]]],
                              results_key: str) -> JSON:
  """Constructs execution data results JSON from a 3d array.

  Args:
    latencies: 3d list of latencies per parameter, plan, and iteration.
    results_key: A string corresponding to the key that maps to the result data.

  Returns:
    JSON with the proper execution results format.
  """
  results = {}
  for i in range(len(latencies)):
    results[f"param{i}"] = {
        "results": [[{
            results_key: v
        } for v in l] for l in latencies[i]]
    }

  return results


def _contains_merge_join(query: str) -> bool:
  return "MergeJoin" in query


def _contains_hash_join(query: str) -> bool:
  return "HashJoin" in query


class PgExecuteTrainingDataQueriesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.templates = {
        test_util.TEST_QUERY_ID: json.loads(test_util.TEST_TEMPLATE_STRING)
    }
    self.hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS)}
    self.values = {test_util.TEST_QUERY_ID: json.loads(_VALUES)}
    self.plan_checks = [_contains_merge_join, _contains_hash_join]
    self.params = [
        value["params"] for value in self.values[test_util.TEST_QUERY_ID]
    ]
    self.keys = [
        "####".join([str(element)
                     for element in param])
        for param in self.params
    ]

    self.results = []
    self.metadata = []
    self.latency = 50

    def checkpoint_results(query_id: str, results: Any,
                           is_metadata: bool) -> None:
      self.assertEqual(test_util.TEST_QUERY_ID, query_id)

      if is_metadata:
        self.metadata.append(copy.deepcopy(results))
      else:
        self.results.append(copy.deepcopy(results))

    self.checkpoint_results = checkpoint_results

    self.query_executions = []

    def execute_query(
        query: str, params: List[Any],
        timeout_ms: Optional[float]) -> Tuple[Optional[float], Optional[int]]:
      """Mocks execution by logging the requested query and returning a latency.

      The query latency changes every call and is boosted by a multiplier based
      on the default plan index so that the behaviors around execution timeouts
      can be verified.

      Args:
        query: SQL query template string with 0 or more parameters provided in
          the form of @param#, starting with 0.
        params: List of parameter values to substitute into the query. All
          values will be cast to str.
        timeout_ms: The statement timeout for this query in ms.

      Returns:
        Tuple:
          1. The execution time of the query in ms or None if the query times
            out.
          2. The number of rows produced by the query or None if the query times
            out.
      """

      self.query_executions.append(QueryExecution(query=query, params=params))
      self.latency -= 1

      multiplier = 1
      for i, plan_check in enumerate(self.plan_checks):
        if plan_check(query):
          multiplier = pow(10, i)
          break

      total_latency = self.latency * multiplier

      if timeout_ms and total_latency >= timeout_ms:
        return None, None

      return total_latency, len(params[1])

    self.execute_query = execute_query

  def _create_simple_execution_orders(
      self, default_plan_indices: List[int]) -> List[List[int]]:
    """Returns an execution order sorted by index except with default first."""
    execution_orders = []
    for default_plan_index in default_plan_indices:
      execution_order = list(range(len(self.hints[test_util.TEST_QUERY_ID])))
      execution_order.remove(default_plan_index)
      execution_order.insert(0, default_plan_index)
      execution_orders.append(execution_order)
    return execution_orders

  def _verify_query_executions(self, iterations: int,
                               params: List[List[str]]) -> None:
    # The order of executions varies because the default plan is always executed
    # first for a given parameter. We ensure that each plan is executed the same
    # number of times.
    plan_check_counts = collections.defaultdict(int)
    for query_execution, (current_params, _, _) in zip(
        self.query_executions,
        itertools.product(params, range(len(self.plan_checks)),
                          range(iterations))):
      self.assertEqual(query_execution.params, current_params)

      for i, plan_check in enumerate(self.plan_checks):
        if plan_check(query_execution.query):
          plan_check_counts[i] += 1
          break

    # Verify that each plan was executed the same number of times.
    self.assertLen(set(plan_check_counts.values()), 1)

  def _verify_results(self,
                      iterations: int,
                      expected_checkpoint_count: int,
                      keys_per_checkpoint: List[List[str]],
                      defaults_per_checkpoint: List[List[int]],
                      execution_orders_per_checkpoint: List[List[List[int]]],
                      rows_per_checkpoint: List[List[int]],
                      results_key: str,
                      skip_indices: Optional[List[int]] = None) -> None:
    if not skip_indices:
      skip_indices = []
    self.assertLen(self.results, expected_checkpoint_count)
    self.assertLen(self.metadata, expected_checkpoint_count)
    self.assertLen(keys_per_checkpoint, expected_checkpoint_count)
    self.assertLen(defaults_per_checkpoint, expected_checkpoint_count)
    self.assertLen(execution_orders_per_checkpoint, expected_checkpoint_count)
    self.assertLen(rows_per_checkpoint, expected_checkpoint_count)

    for result, metadata, current_keys, current_defaults, current_execution_orders, current_rows in zip(
        self.results, self.metadata, keys_per_checkpoint,
        defaults_per_checkpoint, execution_orders_per_checkpoint,
        rows_per_checkpoint):
      self.assertEmpty(metadata[test_util.TEST_QUERY_ID])
      result_map = result[test_util.TEST_QUERY_ID]
      self.assertEqual(len(result_map), len(current_keys))
      for key, default, execution_order, rows in zip(current_keys,
                                                     current_defaults,
                                                     current_execution_orders,
                                                     current_rows):
        self.assertEqual(result_map[key]["default"], default)
        self.assertEqual(result_map[key]["execution_order"], execution_order)
        self.assertEqual(result_map[key]["rows"], rows)
        self.assertLen(result_map[key]["results"], 2)
        for i, stats in enumerate(result_map[key]["results"]):
          if i in skip_indices and i != default:
            self.assertLen(stats, 1)
            self.assertEqual(stats[0], {"skipped": True})
          else:
            self.assertLen(stats, iterations)
            for iteration_entry in stats:
              self.assertGreater(iteration_entry[results_key], 0)

  def _verify_timeouts(self, iterations: int,
                       expected_timeouts: List[bool]) -> None:
    """Ensures that the expected (parameter, plan) pairs have time outs.

    Args:
      iterations: The number of iterations executed per (parameter, plan).
      expected_timeouts: A list with an entry for each (parameter, plan) pair
        being checked signifying whether it should have timed out. The ordering
        must match iterating over keys and for each plan per key.
    """

    self.assertLen(self.metadata, 1)
    self.assertLen(self.results, 1)
    result_map = self.results[0][test_util.TEST_QUERY_ID]
    hints_list = self.hints[test_util.TEST_QUERY_ID]
    assert len(result_map) * len(hints_list) == len(expected_timeouts)

    for expected_timeout, (key, plan_index) in zip(
        expected_timeouts, itertools.product(self.keys,
                                             range(len(hints_list)))):
      plan_entry = result_map[key]["results"][plan_index]
      self.assertLen(plan_entry, iterations)
      for iteration_entry in plan_entry:
        if expected_timeout:
          self.assertIn("timed_out", iteration_entry)
          self.assertNotIn("duration_ms", iteration_entry)
        else:
          self.assertNotIn("timed_out", iteration_entry)
          self.assertIn("duration_ms", iteration_entry)

  @parameterized.named_parameters(
      dict(testcase_name="basic", iterations=1, limit=None),
      dict(testcase_name="iterations_3", iterations=3, limit=None),
      dict(testcase_name="limit_2", iterations=1, limit=2))
  def test_iterations_and_limit(self, iterations, limit):
    """Verifies execution combinations abide by iterations and limit.

    The test checks all the queries that would have been executed as well as the
    shape of the results.

    Args:
      iterations: The number of times to execute the requested query.
      limit: The number of parameter_values to gather execution data for.
    """
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=iterations,
        batch_size=100,
        limit=limit,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    self._verify_query_executions(
        iterations=iterations, params=self.params[:limit])

    keys = self.keys[:limit]
    defaults = [
        int(value["plan_index"])
        for value in self.values[test_util.TEST_QUERY_ID]
    ]
    execution_orders = self._create_simple_execution_orders(defaults)
    rows = [len(params[1]) for params in self.params]
    self._verify_results(
        iterations=iterations,
        expected_checkpoint_count=1,
        keys_per_checkpoint=[keys],
        defaults_per_checkpoint=[defaults],
        execution_orders_per_checkpoint=[execution_orders],
        rows_per_checkpoint=[rows],
        results_key=results_key)

  @parameterized.named_parameters(
      dict(testcase_name="tiny_batch", batch_size=1),
      dict(testcase_name="small_batch", batch_size=2),
      dict(testcase_name="input_size_batch", batch_size=3),
      dict(testcase_name="large_batch", batch_size=1000))
  def test_batch_size(self, batch_size):
    """Verifies results are checkpointed in accordance with batch size.

    The test checks all the queries that would have been executed as well as the
    shape of the results.

    Args:
      batch_size: The number of parameter_values to fully evaluate before
        calling checkpoint_results_fn.
    """
    results_key = "results"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=1,
        batch_size=batch_size,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    self._verify_query_executions(iterations=1, params=self.params)

    defaults = [
        int(value["plan_index"])
        for value in self.values[test_util.TEST_QUERY_ID]
    ]
    expected_checkpoint_count = math.ceil(len(defaults) / batch_size)

    # Handle an edge condition where the last execution iteration will
    # checkpoint immediately before the post-execution checkpoint.
    if len(defaults) % batch_size == 0:
      expected_checkpoint_count += 1

    keys_per_checkpoint = [
        self.keys[:(batch_end + 1) * batch_size]
        for batch_end in range(expected_checkpoint_count)
    ]
    defaults_per_checkpoint = [
        defaults[:(default_end + 1) * batch_size]
        for default_end in range(expected_checkpoint_count)
    ]
    execution_orders_per_checkpoint = [
        self._create_simple_execution_orders(defaults)
        for defaults in defaults_per_checkpoint
    ]
    rows = [len(params[1]) for params in self.params]
    rows_per_checkpoint = [
        rows[:(rows_end + 1) * batch_size]
        for rows_end in range(expected_checkpoint_count)
    ]

    self._verify_results(
        iterations=1,
        expected_checkpoint_count=expected_checkpoint_count,
        keys_per_checkpoint=keys_per_checkpoint,
        defaults_per_checkpoint=defaults_per_checkpoint,
        execution_orders_per_checkpoint=execution_orders_per_checkpoint,
        rows_per_checkpoint=rows_per_checkpoint,
        results_key=results_key)

  @parameterized.named_parameters(
      dict(testcase_name="skip_none", skip_indices=[]),
      dict(testcase_name="skip_0", skip_indices=[0]))
  def test_skip_indices(self, skip_indices):
    """Verifies that execution skips appropriate plans.

    Args:
      skip_indices: List of plan indices to skip.
    """
    # The default plan cannot be skipped. Update the default plans indices to
    # all be 1.
    for value in self.values[test_util.TEST_QUERY_ID]:
      value["plan_index"] = 1

    iterations = 2
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=skip_indices,
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    defaults = [
        int(value["plan_index"])
        for value in self.values[test_util.TEST_QUERY_ID]
    ]
    execution_orders = self._create_simple_execution_orders(defaults)
    rows = [len(params[1]) for params in self.params]
    self._verify_results(
        iterations=iterations,
        expected_checkpoint_count=1,
        keys_per_checkpoint=[self.keys],
        defaults_per_checkpoint=[defaults],
        execution_orders_per_checkpoint=[execution_orders],
        rows_per_checkpoint=[rows],
        results_key=results_key,
        skip_indices=skip_indices)

  def test_timeout_value(self):
    """Verifies that the timeout_ms is set correctly per parameter."""
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=5,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    # The timeout_ms will be the median execution time per parameter.
    self.assertLen(self.results, 1)
    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertEqual(len(result_map), len(self.keys))
    # Explanation of expected_timeouts:
    # 1. The latency in execute_query starts at 49 and reduces by 1 each
    # execution.
    # 2. This test runs 5 iterations per plan over 2 plans and the default plan
    # is always executed first. Therefore median latencies per default plan end
    # in 7.
    # 3. The latency is multiplied by 10^default_plan_index, so the latency when
    # the default plan is 1 is 10x higher.
    # 4. The values described by (1,2,3) are multiplied by
    # _QUERY_TIMEOUT_MULTIPLIER.
    expected_timeouts_ms = [470, 3700, 270]
    for key, expected_timeout_ms in zip(self.keys, expected_timeouts_ms):
      self.assertEqual(result_map[key]["timeout_ms"], expected_timeout_ms)

  def test_timeouts_low_query_timeout_multiplier(self):
    """Verifies that lowering the query timeout multiplier causes timeouts."""

    iterations = 5
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=1,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertEqual(len(result_map), len(self.keys))
    # Expect the entries for the second plan to timeout when the
    # default_plan_index is 0. This occurs for the first and third set of
    # parameters.
    self._verify_timeouts(
        iterations=iterations,
        expected_timeouts=[False, True, False, False, False, True])

  def test_timeouts_low_query_timeout_max(self):
    """Verifies that lowering the query timeout max causes timeouts."""

    iterations = 5
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MIN_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertEqual(len(result_map), len(self.keys))
    # The default plan execution never times out by construction. Everything
    # else will timeout with a low query timeout max.
    self._verify_timeouts(
        iterations=iterations,
        expected_timeouts=[False, True, True, False, False, True])

  def test_timeouts_high_query_timeout_min(self):
    """Verifies that raising the query timeout min removes timeouts."""

    iterations = 5
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=1,
        query_timeout_min_ms=_QUERY_TIMEOUT_MAX_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=self.execute_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertEqual(len(result_map), len(self.keys))
    # Nothing times out with a high query timeout min. This is contrasted with
    # the results of test_timeouts_low_query_timeout_multiplier, where they are
    # query time outs when the query_timeout_multiplier set to 1 and the
    # query_timeout_min is low.
    self._verify_timeouts(
        iterations=iterations,
        expected_timeouts=[False, False, False, False, False, False])

  def _verify_non_first_execution_time_out(self, results: List[List[JSON]],
                                           timed_out_plans: List[int]) -> None:
    for plan_index, plan_results in enumerate(results):
      if plan_index in timed_out_plans:
        self.assertIn("duration_ms", plan_results[0])
        self.assertNotIn("timed_out", plan_results[0])
        for iteration_entry in plan_results[1:]:
          self.assertIn("timed_out", iteration_entry)
          self.assertNotIn("duration_ms", iteration_entry)
      else:
        for iteration_entry in plan_results:
          self.assertIn("duration_ms", iteration_entry)
          self.assertNotIn("timed_out", iteration_entry)

  @parameterized.named_parameters(
      dict(
          testcase_name="no time outs",
          minimum_speedup_multiplier=1,
          expected_timed_out_plans=[]),
      dict(
          testcase_name="slower non-default plan times out",
          minimum_speedup_multiplier=20,
          expected_timed_out_plans=[2]),
      dict(
          testcase_name="both non-default plans time out",
          minimum_speedup_multiplier=60,
          expected_timed_out_plans=[1, 2]))
  def test_timeouts_high_query_timeout_minimum_speedup_multiplier(
      self, minimum_speedup_multiplier: int,
      expected_timed_out_plans: List[int]):
    """Verifies that raising the minimum speed up multiplier causes timeouts."""

    iterations = 5
    results_key = "duration_ms"
    param_values = [{"params": [50, 10, 30], "plan_index": "0"}]

    # The timeout computation multiplied the default execution time by
    # query_timeout_multiplier and divides by
    # query_timeout_minimum_speedup_multiplier. For example, if
    # _QUERY_TIMEOUT_MULTIPLIER=10, then minimum_speedup_multiplier needs to be
    # 20 to halve the default execution time in the timeout computation.
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values={test_util.TEST_QUERY_ID: param_values},
        plan_hints={test_util.TEST_QUERY_ID: _PARAM_HINTS},
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=_execute_hinted_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key,
        query_timeout_minimum_speedup_multiplier=minimum_speedup_multiplier)

    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertLen(result_map, 1)
    results = result_map[next(iter(result_map))]
    self._verify_non_first_execution_time_out(results["results"],
                                              expected_timed_out_plans)

  def test_execute_without_rows(self):
    """Verifies returning None can be returned for number of rows produced."""

    def execute_query_without_rows(*_) -> Tuple[str, None]:
      return "hello", None

    iterations = 3
    results_key = "test"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=self.hints,
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=execute_query_without_rows,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertLen(result_map, 3)
    for results in result_map.values():
      self.assertIsNone(results["rows"])

  def test_adaptive_timeouts(self):
    """Verifies adaptive timeouts based on fast optimal execution times.

    We mock the following execution scenario using a combination of
    execute_query_adaptive_mock and the "hints" values. We also double the
    latency of the first execution to help show that timeouts are computed using
    a median vs a max/mean.

    We do this for the first parameter only and return a consistent latency of
    50 for all other (parameter, plan) pairs. This demonstrates that the timeout
    calculation policy is independent between parameter values.

    1. Default query executes in 21 ms, setting the default_timeout_ms to 210.
    2. The second plan times out under default_timeout_ms.
    3. The third plan executes in 2 ms, setting an optimal_timeout_ms to 20.
    4. The fourth plan times out under the optimal_timeout_ms. Recall the
       first iteration is only held to default_timeout_ms.
    5. The fifth plan can execute under the optimal_timeout_ms.
    """

    iterations = 3
    results_key = "duration_ms"

    hints = {
        test_util.TEST_QUERY_ID: [{
            "hints": "21|"
        }, {
            "hints": "300|"
        }, {
            "hints": "2|"
        }, {
            "hints": "40|"
        }, {
            "hints": "11|"
        }]
    }

    first_execution_check = set()

    def execute_query_adaptive_mock(
        query: str, params: List[Any],
        timeout_ms: Optional[float]) -> Tuple[Optional[float], Optional[int]]:
      if params != self.params[0]:
        return 50, 500

      plan_bias = int(query.split("|")[0])

      if plan_bias not in first_execution_check:
        first_execution_check.add(plan_bias)
        plan_bias = plan_bias * 2

      if timeout_ms and plan_bias >= timeout_ms:
        return None, None

      return plan_bias, len(params[1])

    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values=self.values,
        plan_hints=hints,
        iterations=iterations,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=execute_query_adaptive_mock,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key)

    keys = [
        "####".join([str(element)
                     for element in param])
        for param in self.params
    ]
    result_map = self.results[0][test_util.TEST_QUERY_ID]
    self.assertLen(result_map, 3)

    # We have special checks for the scenario constructed for the first
    # parameter. The remaining parameters have a different and simple behavior
    # to check.
    special_parameter_results = result_map[keys[0]]
    # Verify the default_timeout_ms set by the default plan execution.
    self.assertEqual(special_parameter_results["timeout_ms"], 210)
    # All first executions are held against default_timeout_ms.
    # Default, the first plan, does not have a timeout.
    expected_durations = [
        [{
            "duration_ms": 42
        }, {
            "duration_ms": 21
        }, {
            "duration_ms": 21
        }],
        # Times out under default_timeout_ms = 210.
        [{
            "timed_out": 210
        }, {
            "timed_out": 210
        }, {
            "timed_out": 210
        }],
        [{
            "duration_ms": 4
        }, {
            "duration_ms": 2
        }, {
            "duration_ms": 2
        }],
        # Times out under optimal_timeout_ms = 20, based on above plan.
        [{
            "duration_ms": 80
        }, {
            "timed_out": 20
        }, {
            "timed_out": 20
        }],
        [{
            "duration_ms": 22
        }, {
            "duration_ms": 11
        }, {
            "duration_ms": 11
        }]
    ]
    self.assertEqual(special_parameter_results["results"], expected_durations)
    # The first execution order has no priors.
    self.assertEqual(special_parameter_results["execution_order"],
                     [0, 1, 2, 3, 4])
    expected_execution_orders = [
        # Default is 1 so it's first even though its timed_out runtime is
        # 210ms. 2 is the fastest, then 4. 3 sorts after 0 because it has a time
        # out and so must use the first execution's 80ms when compared to 0's
        # 21ms.
        [1, 2, 4, 0, 3],
        # Default is 0. The remaining plans maintain the relative sort order
        # from the previous iteration (ie, above) since they increased by the
        # same constant. Plan 1's rightful place, when not default, is last
        # because it timed out so badly before.
        [0, 2, 4, 3, 1]
    ]
    for other_parameters, expected_execution_order in zip(
        keys[1:], expected_execution_orders):
      other_parameter_results = result_map[other_parameters]
      self.assertEqual(other_parameter_results["timeout_ms"], 500)
      self.assertEqual(other_parameter_results["execution_order"],
                       expected_execution_order)
      for execution_results in other_parameter_results["results"]:
        self.assertLen(execution_results, iterations)
        for entry in execution_results:
          self.assertEqual(entry["duration_ms"], 50)

  def test_plan_execution_order_manager(self):
    """Verify the execution order creation rules."""

    execution_order_manager = pg_execute_training_data_queries.PlanExecutionOrderManager(
        plan_count=4)
    # The first ordering is simply 0, 1, ... n, except that the default plan
    # comes first.
    self.assertEqual(
        execution_order_manager.get_execution_order_and_reset(
            default_plan_index=2), [2, 0, 1, 3])

    # The order they are added doesn't matter.
    execution_order_manager.add_execution(plan_index=0, execution_latency_ms=10)
    execution_order_manager.add_execution(plan_index=1, execution_latency_ms=12)
    execution_order_manager.add_execution(plan_index=2, execution_latency_ms=4)
    execution_order_manager.add_execution(plan_index=3, execution_latency_ms=7)
    self.assertEqual(
        execution_order_manager.get_execution_order_and_reset(
            default_plan_index=1), [1, 2, 3, 0])

    execution_order_manager.add_execution(plan_index=1, execution_latency_ms=12)
    execution_order_manager.add_execution(plan_index=3, execution_latency_ms=7)
    execution_order_manager.add_execution(plan_index=2, execution_latency_ms=4)
    execution_order_manager.add_execution(plan_index=0, execution_latency_ms=10)
    self.assertEqual(
        execution_order_manager.get_execution_order_and_reset(
            default_plan_index=1), [1, 2, 3, 0])

    # Adding a slow plan changes the order and the default is excused.
    execution_order_manager.add_execution(plan_index=1, execution_latency_ms=12)
    execution_order_manager.add_execution(plan_index=3, execution_latency_ms=7)
    execution_order_manager.add_execution(
        plan_index=2, execution_latency_ms=400)
    execution_order_manager.add_execution(
        plan_index=0, execution_latency_ms=100)
    self.assertEqual(
        execution_order_manager.get_execution_order_and_reset(
            default_plan_index=0), [0, 3, 1, 2])

  def test_plan_execution_order_manager_invariants(self):
    """Verify the execution order creation rules."""

    execution_order_manager = pg_execute_training_data_queries.PlanExecutionOrderManager(
        plan_count=2)

    # Executions cannot be added until after the first reset.
    with self.assertRaisesRegex(AssertionError,
                                "Adding execution for an unknown plan index"):
      execution_order_manager.add_execution(
          plan_index=0, execution_latency_ms=99)
    execution_order_manager.get_execution_order_and_reset(default_plan_index=0)

    # Attempt adding an illegal plan index.
    with self.assertRaisesRegex(AssertionError,
                                "Adding execution for an unknown plan index"):
      execution_order_manager.add_execution(
          plan_index=5, execution_latency_ms=99)

    # Add execution data twice for a plan index.
    execution_order_manager.add_execution(plan_index=0, execution_latency_ms=99)
    with self.assertRaisesRegex(AssertionError,
                                "an execution already been added"):
      execution_order_manager.add_execution(
          plan_index=0, execution_latency_ms=99)

    # Attempt getting the execution order before all plans have execution data
    # for this iteration.
    with self.assertRaisesRegex(AssertionError,
                                "Some plans do not yet have execution data"):
      execution_order_manager.get_execution_order_and_reset(
          default_plan_index=0)

    # Add a None latency for checks after reset.
    execution_order_manager.add_execution(
        plan_index=1, execution_latency_ms=None)
    self.assertEqual(
        execution_order_manager.get_execution_order_and_reset(
            default_plan_index=0), [0, 1])

    with self.assertRaisesRegex(AssertionError,
                                "had non-null result after null"):
      execution_order_manager.add_execution(
          plan_index=1, execution_latency_ms=1)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_loose_near_optimal",
          estimator=database_simulator.LatencyEstimator.MIN,
          near_optimal_threshold=1.1,
          num_params_threshold=1.,
          num_params_limit=6,
          expected_cover={0: 6}),
      dict(
          testcase_name="test_near_optimal",
          estimator=database_simulator.LatencyEstimator.MIN,
          near_optimal_threshold=1.02,
          num_params_threshold=1.,
          num_params_limit=6,
          expected_cover={
              2: 4,
              1: 2
          }),
      dict(
          testcase_name="test_strict_optimal",
          estimator=database_simulator.LatencyEstimator.MIN,
          near_optimal_threshold=1.001,
          num_params_threshold=1.,
          num_params_limit=6,
          expected_cover={
              2: 3,
              1: 2,
              0: 1
          }),
      dict(
          testcase_name="test_params_threshold",
          estimator=database_simulator.LatencyEstimator.MIN,
          near_optimal_threshold=1.001,
          num_params_threshold=0.8,
          num_params_limit=6,
          expected_cover={
              2: 3,
              1: 2
          }),
      dict(
          testcase_name="test_num_params_limit",
          estimator=database_simulator.LatencyEstimator.MIN,
          near_optimal_threshold=1.001,
          num_params_threshold=0.8,
          num_params_limit=4,
          expected_cover={
              2: 3,
              1: 1
          }),
      dict(
          testcase_name="test_num_params_estimator",
          estimator=database_simulator.LatencyEstimator.MAX,
          near_optimal_threshold=1.001,
          num_params_threshold=1.,
          num_params_limit=6,
          expected_cover={
              2: 6,
          }),
  )
  def test_greedy_plan_cover(self, estimator, near_optimal_threshold,
                             num_params_threshold, num_params_limit,
                             expected_cover):
    results_key = "duration_ms"
    fake_execution_times = [[[100, 10], [100, 10.5], [10, 9.5]],
                            [[100, 10], [100, 10.5], [10, 9.5]],
                            [[100, 10], [100, 10.5], [10, 9.5]],
                            [[100, 10], [100, 9.5], [11, 10.5]],
                            [[100, 10], [100, 9.5], [11, 10.5]],
                            [[100, 10], [100, 10.1], [11, 10.1]]]
    execution_results = _construct_execution_data(fake_execution_times,
                                                  results_key)
    plan_cover = pg_execute_training_data_queries.get_greedy_plan_cover(
        execution_results, results_key, estimator, near_optimal_threshold,
        num_params_threshold, num_params_limit)
    self.assertEqual(len(plan_cover), len(expected_cover))
    for plan_idx, num_near_optimal in plan_cover.items():
      self.assertEqual(num_near_optimal, expected_cover[plan_idx])

  @parameterized.named_parameters(
      dict(
          testcase_name="sample_4_3_2",
          num_initial_default_executions=4,
          slowest_default_top_k=3,
          slowest_default_sample_size=2,
          expected_reorder=[3, 2, 0, 1, 4, 5, 6, 7]),
      dict(
          testcase_name="sample_5_3_2",
          num_initial_default_executions=5,
          slowest_default_top_k=3,
          slowest_default_sample_size=2,
          expected_reorder=[4, 3, 0, 1, 2, 5, 6, 7]),
      dict(
          testcase_name="sample_5_1_1",
          num_initial_default_executions=5,
          slowest_default_top_k=1,
          slowest_default_sample_size=1,
          expected_reorder=[4, 0, 1, 2, 3, 5, 6, 7]),
      dict(
          testcase_name="sample_5_2_1",
          num_initial_default_executions=5,
          slowest_default_top_k=2,
          slowest_default_sample_size=1,
          expected_reorder=[4, 0, 1, 2, 3, 5, 6, 7]),
      dict(
          testcase_name="sample_9_9_9",
          num_initial_default_executions=9,
          slowest_default_top_k=9,
          slowest_default_sample_size=9,
          expected_reorder=[0, 5, 4, 3, 6, 1, 2, 7]),
  )
  def test_default_latency_reordering(self, num_initial_default_executions,
                                      slowest_default_top_k,
                                      slowest_default_sample_size,
                                      expected_reorder):
    """Tests reordering by default latencies.

    Args:
      num_initial_default_executions: Number of params to execute default plan
        for, to determine which params to move to the front..
      slowest_default_top_k: Specifies how many of the slowest parameters to
        sample from.
      slowest_default_sample_size: How many of the slowest k parameters to
        sample.
      expected_reorder: The expected order of the params after default-based
        reordering, by original param indices.
    """
    results_key = "duration_ms"
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values={test_util.TEST_QUERY_ID: _PARAM_VALUES},
        plan_hints={test_util.TEST_QUERY_ID: _PARAM_HINTS},
        iterations=2,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=_execute_hinted_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key,
        num_initial_default_executions=num_initial_default_executions,
        slowest_default_top_k=slowest_default_top_k,
        slowest_default_sample_size=slowest_default_sample_size,
        plan_cover_num_params=None,
        near_optimal_threshold=None,
        num_params_threshold=None,
        seed=20)

    self.assertLen(self.results, 1)
    execution_results = self.results[0][test_util.TEST_QUERY_ID]
    self.assertEqual(len(execution_results), len(_PARAM_VALUES))
    for i, (params, param_results) in enumerate(execution_results.items()):
      # Check that we moved slowest sampled queries into the front.
      self.assertEqual(
          params,
          "####".join(map(str, _PARAM_VALUES[expected_reorder[i]]["params"])))

      # Check that everything was executed.
      for plan_results in param_results:
        self.assertTrue(all(["skipped" not in d for d in plan_results]))

  @parameterized.named_parameters(
      dict(
          testcase_name="cover_size1",
          plan_cover_num_params=2,
          params_reorder=[3, 2, 0, 1, 4, 5, 6, 7],
          expected_cover=[1]),
      dict(
          testcase_name="cover_size2",
          plan_cover_num_params=3,
          params_reorder=[3, 2, 0, 1, 4, 5, 6, 7],
          expected_cover=[1, 0]),
      dict(
          testcase_name="cover_size3",
          plan_cover_num_params=4,
          params_reorder=[4, 3, 0, 1, 2, 5, 6, 7],
          expected_cover=[0, 2, 1]),
      dict(
          testcase_name="no_reorder",
          plan_cover_num_params=4,
          params_reorder=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_cover=[0, 1]),
  )
  def test_plan_cover_pruning(self, plan_cover_num_params, params_reorder,
                              expected_cover):
    """Tests greedy plan cover pruning.

    This test shows three cases, with greedy cover sizes 1, 2, and 3. It also
    simulates default latency reordering via params_reorder.

    Args:
      plan_cover_num_params: How many params to compute the plan cover using.
      params_reorder: Executes the parameters in this order to simulate default
        latency reordering.
      expected_cover: The expected plan indices in the plan cover.
    """
    results_key = "duration_ms"
    reordered_values = [_PARAM_VALUES[i] for i in params_reorder]
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values={test_util.TEST_QUERY_ID: reordered_values},
        plan_hints={test_util.TEST_QUERY_ID: _PARAM_HINTS},
        iterations=2,
        batch_size=100,
        limit=None,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=_execute_hinted_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key,
        num_initial_default_executions=None,
        slowest_default_top_k=None,
        slowest_default_sample_size=None,
        plan_cover_num_params=plan_cover_num_params,
        near_optimal_threshold=1.05,
        num_params_threshold=0.99,
        seed=20)

    self.assertLen(self.results, 1)
    execution_results = self.results[0][test_util.TEST_QUERY_ID]
    self.assertEqual(len(execution_results), len(_PARAM_VALUES))
    for i, (params, param_results) in enumerate(execution_results.items()):
      self.assertEqual(params,
                       "####".join(map(str, reordered_values[i]["params"])))
      self.assertLen(param_results["results"], 3)
      # Prior to computing plan cover, we don't skip any plans.
      if i < plan_cover_num_params:
        for plan_results in param_results:
          self.assertTrue(all(["skipped" not in d for d in plan_results]))
      else:
        # After plan_cover_num_params, we skip any non-default not in cover.
        default_index = param_results["default"]
        for plan_index, plan_results in enumerate(param_results["results"]):
          for iteration_results in plan_results:
            if plan_index not in [default_index] + expected_cover:
              self.assertIn("skipped", iteration_results)
            else:
              self.assertNotIn("skipped", iteration_results)

    self.assertLen(self.metadata, 1)
    self.assertEqual(
        self.metadata[0][test_util.TEST_QUERY_ID][
            pg_execute_training_data_queries.PLAN_COVER], expected_cover)

  @parameterized.named_parameters(
      dict(
          testcase_name="timeout a single parameter",
          minimum_speedup_multiplier=20,
          expected_cover=[1, 2]),
      dict(
          testcase_name="timeouts eliminate plan 2",
          minimum_speedup_multiplier=40,
          expected_cover=[1, 0]),
      dict(
          testcase_name="timeouts affect all non-defaults",
          minimum_speedup_multiplier=60,
          expected_cover=[0]),
  )
  def test_plan_cover_pruning_skip_timeouts(self,
                                            minimum_speedup_multiplier: int,
                                            expected_cover: List[int]):
    """Verifies that the plan cover skips (params, plan) with timeouts."""

    iterations = 3
    results_key = "duration_ms"
    param_values = [{
        "params": [50, 10, 20],
        "plan_index": "0"
    }, {
        "params": [50, 20, 15],
        "plan_index": "0"
    }, {
        "params": [50, 30, 30],
        "plan_index": "0"
    }]

    # See test_timeouts_high_query_timeout_minimum_speedup_multiplier regarding
    # how the query_timeout_minimum_speedup_multiplier works. This test uses it
    # to induce time outs in the execution data to test the effect on plan
    # available for the plan cover computation.
    pg_execute_training_data_queries.execute_training_data_queries(
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        parameter_values={test_util.TEST_QUERY_ID: param_values},
        plan_hints={test_util.TEST_QUERY_ID: _PARAM_HINTS},
        iterations=iterations,
        batch_size=100,
        limit=3,
        skip_indices=[],
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS,
        execute_query_fn=_execute_hinted_query,
        checkpoint_results_fn=self.checkpoint_results,
        results_key=results_key,
        num_initial_default_executions=None,
        slowest_default_top_k=None,
        slowest_default_sample_size=None,
        plan_cover_num_params=2,
        near_optimal_threshold=1.05,
        num_params_threshold=0.99,
        query_timeout_minimum_speedup_multiplier=minimum_speedup_multiplier,
        seed=20)

    self.assertLen(self.metadata, 1)
    self.assertEqual(
        self.metadata[0][test_util.TEST_QUERY_ID][
            pg_execute_training_data_queries.PLAN_COVER], expected_cover)


if __name__ == "__main__":
  absltest.main()

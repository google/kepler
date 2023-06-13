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

import copy
import dataclasses
import itertools
import math
from typing import Any, List, Tuple


from kepler.data_management import test_util
from kepler.data_management import workload
from kepler.evaluation import e2e_evaluation

from absl.testing import absltest
from absl.testing import parameterized


JSON = Any


@dataclasses.dataclass
class QueryExecution:
  query: str
  params: List[Any]


class E2EEvaluationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._workload_eval = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED
    ).all()
    self._params = [
        query_instance.parameters
        for query_instance in self._workload_eval.query_log
    ]
    self._keys = [
        "####".join([str(element) for element in param])
        for param in self._params
    ]
    self._template = test_util.QUERY_METADATA[test_util.TEST_QUERY_ID]
    self._results = []

    def checkpoint_results(
        query_id: str,
        results: Any,
    ) -> None:
      self.assertEqual(test_util.TEST_QUERY_ID, query_id)

      self._results.append(copy.deepcopy(results))

    self._checkpoint_results = checkpoint_results

    self._query_executions = []

    self._latency = 1

    def execute_query(query: str, params: List[Any]) -> Tuple[float, int]:
      """Mocks execution by logging the requested query and returning a latency.

      The query latency changes every call.

      Args:
        query: SQL query template string with 0 or more parameters provided in
          the form of @param#, starting with 0.
        params: List of parameter values to substitute into the query. All
          values will be cast to str.

      Returns:
        A tuple containing:
          1. The execution time of the query in ms.
          2. The number of rows produced by the query.
      """
      self._query_executions.append(QueryExecution(query=query, params=params))
      self._latency += 1

      # The 3rd element of a params is an int. Return it as the rows produced.
      return self._latency, params[2]

    self._execute_query = execute_query

  def _verify_query_executions(
      self, iterations: int, params: List[List[str]]
  ) -> None:
    for query_execution, (current_params, _) in zip(
        self._query_executions, itertools.product(params, range(iterations))
    ):
      self.assertIn(f"/*+ {test_util.TEST_QUERY_ID} */", query_execution.query)
      self.assertEqual(query_execution.params, current_params)

  def _verify_results(
      self,
      iterations: int,
      expected_checkpoint_count: int,
      keys_per_checkpoint: List[List[str]],
      rows_per_checkpoint: List[List[int]],
  ) -> None:
    self.assertLen(self._results, expected_checkpoint_count)
    self.assertLen(keys_per_checkpoint, expected_checkpoint_count)
    self.assertLen(rows_per_checkpoint, expected_checkpoint_count)

    for result, current_keys, current_rows in zip(
        self._results, keys_per_checkpoint, rows_per_checkpoint
    ):
      result_map = result[test_util.TEST_QUERY_ID]
      self.assertEqual(len(result_map), len(current_keys))

      for key, rows in zip(current_keys, current_rows):
        self.assertEqual(result_map[key]["default"], 0)
        self.assertEqual(result_map[key]["rows"], rows)
        self.assertLen(result_map[key]["results"], 1)
        stats = result_map[key]["results"][0]
        self.assertLen(stats, iterations)
        for i in range(len(stats) - 1):
          self.assertGreater(
              stats[i + 1]["duration_ms"], stats[i]["duration_ms"]
          )

  @parameterized.named_parameters(
      dict(testcase_name="basic", iterations=1, limit=None),
      dict(testcase_name="iterations_3", iterations=3, limit=None),
      dict(testcase_name="limit_2", iterations=1, limit=2),
  )
  def test_iterations_and_limit(self, iterations, limit):
    """Verifies execution combinations abide by iterations and limit.

    The test checks all the queries that would have been executed as well as the
    shape of the results.

    Args:
      iterations: The number of times to execute the requested query.
      limit: The number of parameter_values to gather execution data for.
    """
    e2e_evaluation.evaluate_workload(
        workload_eval=self._workload_eval,
        template=self._template,
        iterations=iterations,
        batch_size=100,
        limit=limit,
        execute_query_fn=self._execute_query,
        checkpoint_results_fn=self._checkpoint_results,
    )

    self._verify_query_executions(
        iterations=iterations, params=self._params[:limit]
    )

    keys = self._keys[:limit]
    rows = [params[2] for params in self._params]
    self._verify_results(
        iterations=iterations,
        expected_checkpoint_count=1,
        keys_per_checkpoint=[keys],
        rows_per_checkpoint=[rows],
    )

  @parameterized.named_parameters(
      dict(testcase_name="tiny_batch", batch_size=1),
      dict(testcase_name="small_batch", batch_size=2),
      dict(testcase_name="input_size_batch", batch_size=3),
      dict(testcase_name="large_batch", batch_size=1000),
  )
  def test_batch_size(self, batch_size):
    """Verifies results are checkpointed in accordance with batch size.

    The test checks all the queries that would have been executed as well as the
    shape of the results.

    Args:
      batch_size: The number of parameter_values to fully evaluate before
        calling checkpoint_results_fn.
    """
    e2e_evaluation.evaluate_workload(
        workload_eval=self._workload_eval,
        template=self._template,
        iterations=1,
        batch_size=batch_size,
        limit=None,
        execute_query_fn=self._execute_query,
        checkpoint_results_fn=self._checkpoint_results,
    )

    self._verify_query_executions(iterations=1, params=self._params)

    expected_checkpoint_count = math.ceil(len(self._params) / batch_size)

    # Handle an edge condition where the last execution iteration will
    # checkpoint immediately before the post-execution checkpoint.
    if len(self._params) % batch_size == 0:
      expected_checkpoint_count += 1

    keys_per_checkpoint = [
        self._keys[: (batch_end + 1) * batch_size]
        for batch_end in range(expected_checkpoint_count)
    ]
    rows = [params[2] for params in self._params]
    rows_per_checkpoint = [
        rows[: (rows_end + 1) * batch_size]
        for rows_end in range(expected_checkpoint_count)
    ]

    self._verify_results(
        iterations=1,
        expected_checkpoint_count=expected_checkpoint_count,
        keys_per_checkpoint=keys_per_checkpoint,
        rows_per_checkpoint=rows_per_checkpoint,
    )


if __name__ == "__main__":
  absltest.main()

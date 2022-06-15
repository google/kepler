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

"""Tests for DatabaseSimulator."""

import copy

from kepler.data_management import database_simulator
from kepler.data_management import test_util
from kepler.data_management import workload
from absl.testing import absltest
from absl.testing import parameterized


class DatabaseSimulatorTest(parameterized.TestCase):

  def test_execute_timed(self):
    """Verifies query execution look-up and database usage stats."""

    simulator = database_simulator.DatabaseSimulator(
        test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN)

    planned_query_non_default = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1])
    latency, is_default = simulator.execute_timed(planned_query_non_default)
    self.assertEqual(latency, 33)
    self.assertFalse(is_default)
    self.assertEqual(simulator.execution_count, 1)
    self.assertEqual(simulator.execution_cost_ms, 33)

    planned_query_default = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=0,
        parameters=test_util.PARAMETERS_POOL[1])
    latency, is_default = simulator.execute_timed(planned_query_default)
    self.assertEqual(latency, 11)
    self.assertTrue(is_default)
    self.assertEqual(simulator.execution_count, 2)
    self.assertEqual(simulator.execution_cost_ms, 44)

  def test_median_estimator(self):
    """Verifies query execution latency median estimator."""

    simulator = database_simulator.DatabaseSimulator(
        test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MEDIAN)

    planned_query_non_default = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1])
    latency, is_default = simulator.execute_timed(planned_query_non_default)
    self.assertEqual(latency, 45)
    self.assertFalse(is_default)
    self.assertEqual(simulator.execution_count, 1)
    self.assertEqual(simulator.execution_cost_ms, 45)

  def test_timeout(self):
    """Verifies execution timeout logic."""

    simulator = database_simulator.DatabaseSimulator(
        test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN)

    planned_query_non_default = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=0,
        parameters=test_util.PARAMETERS_POOL[4])
    latency, is_default = simulator.execute_timed(planned_query_non_default)
    self.assertEqual(latency, 31.)
    self.assertFalse(is_default)
    self.assertEqual(simulator.execution_count, 1)
    self.assertEqual(simulator.execution_cost_ms, 31)

    planned_query_default = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[4])
    latency, is_default = simulator.execute_timed(planned_query_default)
    self.assertEqual(latency, 1)
    self.assertTrue(is_default)
    self.assertEqual(simulator.execution_count, 2)
    self.assertEqual(simulator.execution_cost_ms, 32)

  def test_execute_explain(self):
    """Verifies query plan explain total cost look-up."""

    simulator = database_simulator.DatabaseSimulator(
        query_execution_data=test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1])
    total_cost, plan_id = simulator.execute_explain(planned_query)
    self.assertEqual(total_cost, 15.1)
    self.assertEqual(plan_id, 2)

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=0,
        parameters=test_util.PARAMETERS_POOL[3])
    total_cost, plan_id = simulator.execute_explain(planned_query)
    self.assertEqual(total_cost, 33.1)
    self.assertEqual(plan_id, 0)

  @parameterized.named_parameters(("plan_0", 0, 0), ("plan_2", 2, 2),
                                  ("plan_none", None, 0))
  def test_get_plan_id(self, plan_id: int, expected_plan_id: int):
    """Verifies get_plan_id."""

    simulator = database_simulator.DatabaseSimulator(
        query_execution_data=test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=plan_id,
        parameters=test_util.PARAMETERS_POOL[1])
    returned_plan_id = simulator.get_plan_id(planned_query)
    self.assertEqual(returned_plan_id, expected_plan_id)

  def test_execute_explain_not_initialized(self):
    """Verifies query explain look-up fails when data not provided."""

    simulator = database_simulator.DatabaseSimulator(
        test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN)

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1])

    self.assertRaisesRegex(ValueError,
                           "Called execute_explain without providing",
                           simulator.execute_explain, planned_query)

  def test_execute_default_plan(self):
    """Verifies executing the default query plan."""

    simulator = database_simulator.DatabaseSimulator(
        query_execution_data=test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)

    planned_query_default = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=None,
        parameters=test_util.PARAMETERS_POOL[2])

    latency, is_default = simulator.execute_timed(planned_query_default)
    self.assertEqual(latency, 111)
    self.assertTrue(is_default)

    total_cost, default_plan_id = simulator.execute_explain(
        planned_query_default)
    self.assertEqual(total_cost, 24.1)
    self.assertEqual(default_plan_id, 1)

  def test_illegal_init_calls(self):
    """Verifies exceptions for illegal init calls."""

    too_many_execution_keys = copy.deepcopy(test_util.QUERY_EXECUTION_DATA)
    too_many_execution_keys[test_util.TEST_QUERY_ID + "_2"] = (
        too_many_execution_keys[test_util.TEST_QUERY_ID])
    self.assertRaisesRegex(ValueError, "Unexpected data format",
                           database_simulator.DatabaseSimulator,
                           too_many_execution_keys,
                           test_util.QUERY_EXECUTION_METADATA,
                           database_simulator.LatencyEstimator.MIN)

    too_many_metadata_keys = copy.deepcopy(test_util.QUERY_EXECUTION_METADATA)
    too_many_metadata_keys[test_util.TEST_QUERY_ID + "_2"] = (
        too_many_metadata_keys[test_util.TEST_QUERY_ID])
    self.assertRaisesRegex(ValueError, "Unexpected metadata format",
                           database_simulator.DatabaseSimulator,
                           test_util.QUERY_EXECUTION_DATA,
                           too_many_metadata_keys,
                           database_simulator.LatencyEstimator.MIN)

    too_many_explain_keys = copy.deepcopy(test_util.QUERY_EXPLAIN_DATA)
    too_many_explain_keys[test_util.TEST_QUERY_ID + "_2"] = (
        too_many_explain_keys[test_util.TEST_QUERY_ID])
    self.assertRaisesRegex(ValueError, "Unexpected explain data format",
                           database_simulator.DatabaseSimulator,
                           test_util.QUERY_EXECUTION_DATA,
                           test_util.QUERY_EXECUTION_METADATA,
                           database_simulator.LatencyEstimator.MIN,
                           too_many_explain_keys)

    missing_query_id_metadata = {
        test_util.TEST_QUERY_ID + "_2":
            test_util.QUERY_EXECUTION_METADATA[test_util.TEST_QUERY_ID]
    }
    self.assertRaisesRegex(
        ValueError, "Query id mismatch between data arguments",
        database_simulator.DatabaseSimulator, test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        database_simulator.LatencyEstimator.MIN, missing_query_id_metadata)

    missing_query_id_explain = {
        test_util.TEST_QUERY_ID + "_2":
            test_util.QUERY_EXPLAIN_DATA[test_util.TEST_QUERY_ID]
    }
    self.assertRaisesRegex(
        ValueError, "Query id mismatch between data arguments",
        database_simulator.DatabaseSimulator, test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        database_simulator.LatencyEstimator.MIN, missing_query_id_explain)

    illegal_explains_key = copy.deepcopy(test_util.QUERY_EXPLAIN_DATA)
    first_entry = illegal_explains_key[test_util.TEST_QUERY_ID]
    first_entry_stats = first_entry[next(iter(first_entry))]
    first_entry_stats["explains"] = {}
    self.assertRaisesRegex(ValueError,
                           "Execution data contains key \"explains\"",
                           database_simulator.DatabaseSimulator,
                           illegal_explains_key,
                           test_util.QUERY_EXECUTION_METADATA,
                           database_simulator.LatencyEstimator.MIN)

    missing_parameter = copy.deepcopy(test_util.QUERY_EXPLAIN_DATA)
    first_entry = missing_parameter[test_util.TEST_QUERY_ID]
    del missing_parameter[test_util.TEST_QUERY_ID][next(iter(first_entry))]
    self.assertRaisesRegex(
        ValueError, "query_explain_data must contain all parameter keys found ",
        database_simulator.DatabaseSimulator, test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        database_simulator.LatencyEstimator.MIN, missing_parameter)

    skipped_default_plan = copy.deepcopy(test_util.QUERY_EXECUTION_DATA)
    first_entry = skipped_default_plan[test_util.TEST_QUERY_ID]
    first_entry_stats = first_entry[next(iter(first_entry))]
    first_entry_stats["results"] = [[{
        "skipped": True
    }]] + first_entry_stats["results"][1:]
    self.assertRaisesRegex(ValueError, "Default plan 0 skipped for params",
                           database_simulator.DatabaseSimulator,
                           skipped_default_plan,
                           test_util.QUERY_EXECUTION_METADATA,
                           database_simulator.LatencyEstimator.MIN)

    skipped_plan_cover_plan = copy.deepcopy(test_util.QUERY_EXECUTION_DATA)
    first_entry = skipped_plan_cover_plan[test_util.TEST_QUERY_ID]
    first_entry_stats = first_entry[next(iter(first_entry))]
    first_entry_stats["results"] = first_entry_stats["results"][:1] + [[{
        "skipped": True
    }]] + first_entry_stats["results"][2:]
    self.assertRaisesRegex(ValueError, "Plan id 1 from the plan cover",
                           database_simulator.DatabaseSimulator,
                           skipped_plan_cover_plan,
                           test_util.QUERY_EXECUTION_METADATA,
                           database_simulator.LatencyEstimator.MIN)

  @parameterized.named_parameters(
      dict(
          testcase_name="execute_timed", execute_function_name="execute_timed"),
      dict(
          testcase_name="execute_explain",
          execute_function_name="execute_explain"),
  )
  def test_illegal_queries(self, execute_function_name: str):
    """Verifies exceptions for illegal query explain look-up calls."""
    simulator = database_simulator.DatabaseSimulator(
        query_execution_data=test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)

    wrong_query_id = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID + "_2",
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1])
    self.assertRaisesRegex(ValueError,
                           "Database Simulator is for query template",
                           getattr(simulator,
                                   execute_function_name), wrong_query_id)

    malformed_parameters = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1][1:])
    self.assertRaisesRegex(ValueError,
                           "All parameter bindings must be provided",
                           getattr(simulator,
                                   execute_function_name), malformed_parameters)

    parameters = copy.deepcopy(test_util.PARAMETERS_POOL[1])
    parameters[0] = "mystery"
    out_of_universe_parameters = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID, plan_id=2, parameters=parameters)
    self.assertRaisesRegex(ValueError, "Out-of-universe ",
                           getattr(simulator, execute_function_name),
                           out_of_universe_parameters)

    unknown_plan_id = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=100,
        parameters=test_util.PARAMETERS_POOL[1])
    self.assertRaisesRegex(ValueError, "does not refer to a recognized plan",
                           getattr(simulator, execute_function_name),
                           unknown_plan_id)

  def test_execute_timed_skipped_plan(self):
    """Verifies exceptions for timed execution look-up for skipped plans."""
    simulator = database_simulator.DatabaseSimulator(
        query_execution_data=test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)

    skipped_plan_id = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=3,
        parameters=test_util.PARAMETERS_POOL[2])
    self.assertRaisesRegex(
        ValueError,
        "Cannot execute query. No execution data was provided for plan 3",
        simulator.execute_timed, skipped_plan_id)


class DatabaseClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)
    self._workload = workload_generator.random_sample(2)
    self._database = database_simulator.DatabaseSimulator(
        test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=test_util.QUERY_EXPLAIN_DATA)
    self._database_no_explain = database_simulator.DatabaseSimulator(
        test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=None)
    self._client = database_simulator.DatabaseClient(self._database)
    self._client_no_explain = database_simulator.DatabaseClient(
        self._database_no_explain)

  def test_create_query_batch(self):
    """Verifies creation of the crossproduct of execution requests."""

    plan_ids = [1, 20, 14]
    batch = workload.create_query_batch(plan_ids, self._workload)
    self.assertLen(batch, len(plan_ids) * len(self._workload.query_log))

    expected_batch = []
    for query_instance in self._workload.query_log:
      for plan_id in plan_ids:
        expected_batch.append(
            database_simulator.PlannedQuery(
                query_id=test_util.TEST_QUERY_ID,
                plan_id=plan_id,
                parameters=query_instance.parameters))

    self.assertEqual(batch, expected_batch)

  def test_execute_timed_batch_single(self):
    """Verifies usage of batch interface for singular usage.

    The database usage stats should be updated and accessible.
    """

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=0,
        parameters=test_util.PARAMETERS_POOL[1])
    result = self._client.execute_timed_batch(planned_queries=[planned_query],
                                              get_total_cost=True)

    self.assertEqual(result.shape, (1, 8))
    self.assertEqual(result["param0"][0], "first")
    self.assertEqual(result["param1"][0], "a")
    self.assertEqual(result["param2"][0], "1")
    self.assertEqual(result["param3"][0], "1970-01-02")
    self.assertEqual(result["plan_id"][0], 0)
    self.assertEqual(result["total_cost"][0], 13.1)
    self.assertEqual(result["latency_ms"][0], 11)
    self.assertTrue(result["is_default"][0], 0)

    self.assertEqual(self._database.execution_count, 1)
    self.assertEqual(self._database.execution_cost_ms, 11)

  def test_execute_timed_batch_no_explain(self):
    """Verifies usage of batch interface with no explain data.
    """

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=0,
        parameters=test_util.PARAMETERS_POOL[1])
    result = self._client_no_explain.execute_timed_batch(
        planned_queries=[planned_query],
        get_total_cost=False)

    self.assertEqual(result.shape, (1, 8))
    self.assertEqual(result["param0"][0], "first")
    self.assertEqual(result["param1"][0], "a")
    self.assertEqual(result["param2"][0], "1")
    self.assertEqual(result["param3"][0], "1970-01-02")
    self.assertEqual(result["plan_id"][0], 0)
    self.assertEqual(result["total_cost"][0], -1)
    self.assertEqual(result["latency_ms"][0], 11)
    self.assertTrue(result["is_default"][0], 0)

    self.assertEqual(self._database_no_explain.execution_count, 1)
    self.assertEqual(self._database_no_explain.execution_cost_ms, 11)

  def test_execute_timed_batch_integrated(self):
    """Verifies integrated database client usage.

    This test combines the usage of the KeplerPlanDiscoverer, the
    WorkloadGenerator, and then DatabaseClient (backed by DatabaseSimulator) to
    generate training data for all plan candidates across a Workload.
    """
    plan_ids = workload.KeplerPlanDiscoverer(
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA).plan_ids
    batch = workload.create_query_batch(plan_ids, self._workload)
    results = self._client.execute_timed_batch(planned_queries=batch,
                                               get_total_cost=False)

    self.assertLen(results, len(plan_ids) * len(self._workload.query_log))
    for (_, result), planned_query in zip(results.iterrows(), batch):
      for i, parameter in enumerate(planned_query.parameters):
        self.assertEqual(result[f"param{i}"], parameter)
      self.assertEqual(result["plan_id"], planned_query.plan_id)
      self.assertEqual(result["total_cost"], -1)

    self.assertEqual(self._database.execution_count,
                     len(plan_ids) * len(self._workload.query_log))
    self.assertEqual(self._database.execution_cost_ms,
                     results["latency_ms"].sum())

  def test_illegal_batched_queries(self):
    """Verifies executions for illegal batched execution calls."""

    self.assertRaisesRegex(ValueError, "Cannot execute empty batch",
                           self._client.execute_timed_batch, [])

    planned_query = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=0,
        parameters=test_util.PARAMETERS_POOL[1])
    malformed_parameters = database_simulator.PlannedQuery(
        query_id=test_util.TEST_QUERY_ID,
        plan_id=2,
        parameters=test_util.PARAMETERS_POOL[1][1:])
    self.assertRaisesRegex(ValueError,
                           "All parameter bindings must be provided",
                           self._client.execute_timed_batch,
                           [planned_query, malformed_parameters])


if __name__ == "__main__":
  absltest.main()

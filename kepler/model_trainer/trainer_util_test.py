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

"""Tests for trainer module.
"""
import numpy as np

from kepler.data_management import database_simulator
from kepler.data_management import test_util as data_test_util
from kepler.data_management import workload
from kepler.model_trainer import trainer_util
from absl.testing import absltest
from absl.testing import parameterized

_MAX_MARGINAL_IMPROVEMENT_EXECUTION_DATA = {
    "1####a": {
        "default": 0,
        "results": [[{
            "duration_ms": 10
        }], [{
            "duration_ms": 5
        }]]
    },
    "2####b": {
        "default": 1,
        "results": [[{
            "duration_ms": 10
        }], [{
            "duration_ms": 20
        }]]
    },
    "3####c": {
        "default": 0,
        "results": [[{
            "duration_ms": 5
        }], [{
            "duration_ms": 4
        }]]
    },
    "4####c": {
        "default": 0,
        "results": [[{
            "duration_ms": 100
        }], [{
            "duration_ms": 4
        }]]
    },
    "5####b": {
        "default": 1,
        "results": [[{
            "duration_ms": 10
        }], [{
            "duration_ms": 2000
        }]]
    },
}

_MAX_MARGINAL_IMPROVEMENT_PREDICATE_METADATA = [{
    "alias": "a",
    "column": "col0",
    "operator": "=",
    "data_type": "float"
}, {
    "alias": "b",
    "column": "col1",
    "operator": "=",
    "data_type": "text",
    "distinct_values": ["a", "b", "c"]
}]


class TrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=data_test_util.QUERY_EXECUTION_METADATA)
    self._database = database_simulator.DatabaseSimulator(
        query_execution_data=data_test_util.QUERY_EXECUTION_DATA,
        query_execution_metadata=data_test_util.QUERY_EXECUTION_METADATA,
        estimator=database_simulator.LatencyEstimator.MIN,
        query_explain_data=data_test_util.QUERY_EXPLAIN_DATA)
    self._client = database_simulator.DatabaseClient(self._database)
    self._metadata = data_test_util.QUERY_METADATA[data_test_util.TEST_QUERY_ID]

    self._workload_generator = workload.WorkloadGenerator(
        data_test_util.QUERY_EXECUTION_DATA, data_test_util.TEST_SEED)

  @parameterized.named_parameters(
      ("empty_workload", 0),
      ("sample1", 1),
      ("sample2", 2),
      ("sample3", 3),
  )
  def test_construct_multihead_model_inputs(self, sample_size):
    sampled_workload = self._workload_generator.random_sample(sample_size)
    model_inputs = trainer_util.construct_multihead_model_inputs(
        sampled_workload, self._metadata["predicates"])
    if sample_size == 0:
      self.assertEqual(model_inputs, [])
    else:
      self.assertLen(model_inputs, len(self._metadata["predicates"]))
      for _, (feature_col, predicate) in enumerate(
          zip(model_inputs, self._metadata["predicates"])):
        self.assertLen(feature_col, sample_size)
        # Test that dates are preprocessed to float timestamps.
        if predicate.get("preprocess_type") == "to_timestamp":
          self.assertEqual(feature_col.dtype, np.float64)

  @parameterized.named_parameters(
      ("first1", 1, ["a"]),
      ("first2", 2, ["b", "a"]),
      ("first3", 3, ["b", "a", "c"]),
      ("first4", 4, ["c", "b", "a"]),
      ("first5", 5, ["b", "c", "a"]),
  )
  def test_max_marginal_improvement_vocabulary(self, first_n,
                                               expected_vocabulary):
    execution_data = {}
    for i, (k,
            v) in enumerate(_MAX_MARGINAL_IMPROVEMENT_EXECUTION_DATA.items()):
      if i < first_n:
        execution_data[k] = v
    execution_data = {data_test_util.TEST_QUERY_ID: execution_data}
    execution_metadata = {data_test_util.TEST_QUERY_ID: {"plan_cover": [0, 1]}}
    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=execution_metadata)
    database = database_simulator.DatabaseSimulator(
        query_execution_data=execution_data,
        query_execution_metadata=execution_metadata,
        estimator=database_simulator.LatencyEstimator.MIN)
    client = database_simulator.DatabaseClient(database)
    workload_generator = workload.WorkloadGenerator(execution_data,
                                                    data_test_util.TEST_SEED)

    queries = workload.create_query_batch(plans.plan_ids + [None],
                                          workload_generator.all())
    train_df = client.execute_timed_batch(
        planned_queries=queries, get_total_cost=False)

    vocabulary = trainer_util.get_vocabulary_by_max_marginal_improvement(
        query_execution_df=train_df,
        predicate_metadata=_MAX_MARGINAL_IMPROVEMENT_PREDICATE_METADATA,
        column_index=1)

    self.assertEqual(vocabulary, expected_vocabulary)


if __name__ == "__main__":
  absltest.main()

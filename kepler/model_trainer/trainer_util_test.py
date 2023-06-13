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
import copy
import numpy as np
import tensorflow as tf

from kepler.data_management import database_simulator
from kepler.data_management import test_util as data_test_util
from kepler.data_management import workload
from kepler.model_trainer import evaluation
from kepler.model_trainer import test_util
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

  @parameterized.parameters(
      (1),
      (2),
      (100)
  )
  def test_add_vocabulary_to_metadata(self, vocab_size_limit):
    # Actual vocab values don't matter here.
    fixed_vocab_size = 50
    def _test_get_distinct_values(table, column):
      del table, column
      return [1 for i in range(fixed_vocab_size)]

    queries = workload.create_query_batch(self._plans.plan_ids,
                                          self._workload_generator.all())
    train_df = self._client.execute_timed_batch(
        planned_queries=queries, get_total_cost=False)

    trainer_util.add_vocabulary_to_metadata(
        train_df,
        self._metadata,
        get_distinct_values=_test_get_distinct_values,
        vocab_size_limit=vocab_size_limit)

    for predicate in self._metadata["predicates"]:
      if "distinct_values" in predicate:
        self.assertLessEqual(len(predicate["distinct_values"]),
                             min(vocab_size_limit, fixed_vocab_size))

  @parameterized.parameters(
      (1, 2, 3, 4),
      (5, 6, 7, 8)
  )
  def test_construct_preprocessing_config(self, int_embedding_dim,
                                          int_oov_indices,
                                          text_embedding_dim,
                                          text_oov_indices):
    preprocessing_config = trainer_util.construct_preprocessing_config(
        self._metadata,
        int_embedding_dim,
        int_oov_indices,
        text_embedding_dim,
        text_oov_indices)
    self.assertEqual(len(preprocessing_config),
                     len(self._metadata["predicates"]))
    for pred_metadata, pred_config in zip(self._metadata["predicates"],
                                          preprocessing_config):
      if pred_metadata["data_type"] == "int":
        self.assertEqual(pred_config["output_dim"], int_embedding_dim)
        self.assertEqual(pred_config["num_oov_indices"], int_oov_indices)
      elif pred_metadata["data_type"] == "text":
        self.assertEqual(pred_config["output_dim"], text_embedding_dim)
        self.assertEqual(pred_config["num_oov_indices"], text_oov_indices)

  @parameterized.parameters(
      (1.2, 20, 10, 8),
      (2.2, 25, 100, 5),
  )
  def test_get_sample_weight(self,
                             suboptimality_threshold,
                             suboptimality_weight,
                             improvement_scaling_factor,
                             expected_num_suboptimal):
    queries = workload.create_query_batch(self._plans.plan_ids,
                                          self._workload_generator.all())
    train_df = self._client.execute_timed_batch(
        planned_queries=queries, get_total_cost=False)

    default_latencies = evaluation.get_default_latencies(
        database=self._database,
        query_workload=self._workload_generator.all())
    default_latencies = np.atleast_2d(default_latencies).T

    all_latencies = np.array(train_df["latency_ms"]).reshape(
        (-1, len(self._plans.plan_ids)))
    sample_weight = trainer_util.get_sample_weight(
        all_latencies, default_latencies,
        suboptimality_threshold=suboptimality_threshold,
        suboptimality_weight=suboptimality_weight,
        improvement_scaling_factor=improvement_scaling_factor)

    # Check that suboptimal plans are weighted correctly.
    self.assertEqual(np.sum(sample_weight == suboptimality_weight),
                     expected_num_suboptimal)

    # Check that improvement plans are scaled properly.
    self.assertAlmostEqual(sample_weight[3][1][0],
                           1 + 7.0139 * improvement_scaling_factor,
                           places=1)
    self.assertAlmostEqual(sample_weight[3][2][0],
                           1 + 7.7066 * improvement_scaling_factor,
                           places=1)

  @parameterized.named_parameters(
      ("empty_workload", 0),
      ("sample1", 1),
      ("sample2", 2),
      ("sample3", 3),
  )
  def test_construct_multihead_model_inputs(self, sample_size):
    sampled_workload = self._workload_generator.random_sample(sample_size)
    model_inputs = trainer_util.construct_multihead_model_inputs(
        sampled_workload)
    if sample_size == 0:
      self.assertEqual(model_inputs, [])
    else:
      self.assertLen(model_inputs, len(self._metadata["predicates"]))
      for _, feature_col in enumerate(model_inputs):
        self.assertLen(feature_col, sample_size)

  def test_apply_preprocessing(self):
    x = copy.deepcopy(test_util.TEST_INPUT_PARAMS_0)
    dtypes = [row.dtype for row in x]
    metadata = test_util.TEST_METADATA_0
    self.assertLen(x, len(metadata["predicates"]))
    trainer_util.apply_preprocessing(x, metadata["predicates"])
    for feature_col, dtype, predicate in zip(x, dtypes, metadata["predicates"]):
      # Test that dates are preprocessed to float timestamps.
      if predicate.get("preprocess_type") == "to_timestamp":
        self.assertEqual(feature_col.dtype, np.float64)
        self.assertNotEqual(feature_col.dtype, dtype)
      else:
        self.assertEqual(feature_col.dtype, dtype)

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

  def test_extract_tflite_parameter_index(self):
    model = test_util.ConcreteModelBase(
        test_util.TEST_METADATA_0,
        list(range(test_util.TEST_NUM_PLANS_0)),
        test_util.TEST_MODEL_CONFIG_0,
        test_util.TEST_PREPROCESSING_CONFIG_0,
    )
    interpreter = tf.lite.Interpreter(
        model_content=trainer_util.convert_to_tflite(model.get_model())
    )
    expected_parameter_indexes = [2, 6, 4, 3, 1, 5, 0]
    for expected, input_detail in zip(
        expected_parameter_indexes, interpreter.get_input_details()
    ):
      self.assertEqual(
          expected, trainer_util.extract_tflite_parameter_index(input_detail)
      )


if __name__ == "__main__":
  absltest.main()

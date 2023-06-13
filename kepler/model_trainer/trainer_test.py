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
from typing import Any, List

import numpy as np

from kepler.data_management import database_simulator
from kepler.data_management import test_util as data_test_util
from kepler.data_management import workload
from kepler.model_trainer import multihead_model
from kepler.model_trainer import test_util
from kepler.model_trainer import trainer
from kepler.model_trainer import trainer_util
from absl.testing import absltest
from absl.testing import parameterized


def _construct_query_execution_metadata(plan_cover: List[int]) -> Any:
  return {data_test_util.TEST_QUERY_ID: {"plan_cover": plan_cover}}


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

  def _get_query_execution_df(self,
                              query_execution_metadata,
                              include_default=False):
    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=query_execution_metadata)
    workload_generator = workload.WorkloadGenerator(
        data_test_util.QUERY_EXECUTION_DATA, data_test_util.TEST_SEED)
    full_workload = workload_generator.all()
    plan_ids = plans.plan_ids + [None] if include_default else plans.plan_ids
    queries = workload.create_query_batch(plan_ids, full_workload)

    return self._client.execute_timed_batch(
        planned_queries=queries)

  @parameterized.named_parameters(
      ("plan_cover1", [0, 1, 2],
       np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])),
      ("plan_cover2", [0, 2],
       np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])),
  )
  def test_classification_training_data_plan_cover(self, plan_cover,
                                                   expected_output):
    """Checks that ClassificationTrainer produces correct training data."""
    query_execution_metadata = _construct_query_execution_metadata(plan_cover)
    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=query_execution_metadata)
    query_execution_df = self._get_query_execution_df(query_execution_metadata)

    model = multihead_model.MultiheadModel(
        self._metadata,
        plans.plan_ids,
        test_util.TEST_MODEL_CONFIG_0,
        test_util.TEST_PREPROCESSING_CONFIG_1)
    classification_trainer = trainer.ClassificationTrainer(self._metadata,
                                                           plans, model)

    query_id = data_test_util.TEST_QUERY_ID
    num_params = len(data_test_util.QUERY_EXECUTION_DATA[query_id]) - 1
    x, y = classification_trainer.construct_training_data(query_execution_df)

    # Check that inputs are valid.
    expected_inputs = [
        np.array(["first", "first", "first", "first", "first"]),
        np.array(["a", "a", "b", "b", "c"]),
        np.array([2, 1, 0, 1, 0]),
        np.arange(5) * 86400
    ]
    self.assertLen(x, len(self._metadata["predicates"]))
    for feature, predicate, expected in zip(x, self._metadata["predicates"],
                                            expected_inputs):
      self.assertTrue(np.array_equal(feature, expected))
      self.assertEqual(
          feature.dtype,
          trainer_util.get_np_type(predicate["data_type"]))
      self.assertLen(feature, num_params)

    # Check that targets are valid.
    self.assertTrue(np.array_equal(y, expected_output))

    # Sanity check that the model can train.
    classification_trainer.train(
        x, y, epochs=1, batch_size=len(y), sample_weight=np.arange(len(x[0])))

  @parameterized.named_parameters(
      ("plan_cover1_1.1", [0, 1, 2], 1.1,
       np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])),
      ("plan_cover1_2.1", [0, 1, 2], 2.1,
       np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])),
      ("plan_cover2_1.1", [0, 2], 1.1,
       np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])),
      ("plan_cover2_2.1", [0, 2], 2.1,
       np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])),
      ("plan_cover1_1.001_default", [0, 1, 2], 1.001,
       np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]), True),
      ("plan_cover1_2.1_default", [0, 1, 2], 2.1,
       np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]), True),
      ("plan_cover2_1.1_default", [0, 2], 1.1,
       np.array([[1, 0], [1, 0], [0, 0], [0, 1], [0, 1]]), True),
      ("plan_cover2_2.1_default", [0, 2], 2.1,
       np.array([[1, 0], [1, 0], [0, 0], [0, 1], [0, 1]]), True),
  )
  def test_near_optimal_classification_training_data_plan_cover(
      self,
      plan_cover,
      near_optimal_threshold,
      expected_output,
      default_relative=False):
    """Checks that NearOptimalClassificationTrainer produces correct training data."""
    query_execution_metadata = _construct_query_execution_metadata(plan_cover)
    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=query_execution_metadata)
    query_execution_df = self._get_query_execution_df(query_execution_metadata,
                                                      default_relative)

    model = multihead_model.MultiheadModel(
        self._metadata, plans.plan_ids, test_util.TEST_MODEL_CONFIG_0,
        test_util.TEST_PREPROCESSING_CONFIG_1)
    near_optimal_trainer = trainer.NearOptimalClassificationTrainer(
        self._metadata, plans, model)

    query_id = data_test_util.TEST_QUERY_ID
    num_params = len(data_test_util.QUERY_EXECUTION_DATA[query_id]) - 1
    x, y = near_optimal_trainer.construct_training_data(query_execution_df,
                                                        near_optimal_threshold,
                                                        default_relative)

    # Check that inputs are valid.
    expected_inputs = [
        np.array(["first", "first", "first", "first", "first"]),
        np.array(["a", "a", "b", "b", "c"]),
        np.array([2, 1, 0, 1, 0]),
        np.arange(5) * 86400
    ]
    self.assertLen(x, len(self._metadata["predicates"]))
    for feature, predicate, expected in zip(x, self._metadata["predicates"],
                                            expected_inputs):
      self.assertTrue(np.array_equal(feature, expected))
      self.assertEqual(feature.dtype,
                       trainer_util.get_np_type(predicate["data_type"]))
      self.assertLen(feature, num_params)

    # Check that targets are valid.
    self.assertTrue(np.array_equal(y, expected_output))

    # Sanity check that the model can train.
    near_optimal_trainer.train(
        x, y, epochs=1, batch_size=len(y), sample_weight=np.arange(len(x[0])))

  @parameterized.named_parameters(
      ("plan_cover1", [0, 1, 2],
       np.array([[1, 2, 3], [11, 22, 33], [222, 111, 333],
                 [3333, 2222, 1111], [50, 32, 1]])),
      ("plan_cover2", [0, 2],
       np.array([[1, 3], [11, 33], [222, 333],
                 [3333, 1111], [50, 1]]))
  )
  def test_regression_training_data(self, plan_cover, expected_latencies):
    """Checks that RegressionTrainer produces correct training data."""
    query_execution_metadata = _construct_query_execution_metadata(plan_cover)
    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=query_execution_metadata)
    query_execution_df = self._get_query_execution_df(query_execution_metadata)

    model = multihead_model.MultiheadModel(
        self._metadata,
        plans.plan_ids,
        test_util.TEST_MODEL_CONFIG_2,
        test_util.TEST_PREPROCESSING_CONFIG_1)
    regression_trainer = trainer.RegressionTrainer(self._metadata,
                                                   plans, model)

    query_id = data_test_util.TEST_QUERY_ID
    num_params = len(data_test_util.QUERY_EXECUTION_DATA[query_id]) - 1
    x, y = regression_trainer.construct_training_data(query_execution_df)

    # Check that inputs are valid.
    expected_inputs = [
        np.array(["first", "first", "first", "first", "first"]),
        np.array(["a", "a", "b", "b", "c"]),
        np.array([2, 1, 0, 1, 0]),
        np.arange(5) * 86400
    ]
    self.assertLen(x, len(self._metadata["predicates"]))
    for feature, predicate, expected in zip(x, self._metadata["predicates"],
                                            expected_inputs):
      self.assertTrue(np.array_equal(feature, expected))
      self.assertEqual(
          feature.dtype,
          trainer_util.get_np_type(predicate["data_type"]))
      self.assertLen(feature, num_params)

    # Check that targets are valid.
    self.assertTrue(np.array_equal(y, expected_latencies))

    # Sanity check that the model can train.
    regression_trainer.train(x, y, epochs=1, batch_size=len(y))

  def test_model_history(self):
    """Checks that training history contains specified metrics."""
    plan_cover = list(range(3))
    query_execution_metadata = _construct_query_execution_metadata(plan_cover)
    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=query_execution_metadata)
    query_execution_df = self._get_query_execution_df(query_execution_metadata)

    model = multihead_model.MultiheadModel(
        self._metadata,
        plans.plan_ids,
        test_util.TEST_MODEL_CONFIG_2,
        test_util.TEST_PREPROCESSING_CONFIG_1)
    regression_trainer = trainer.RegressionTrainer(self._metadata,
                                                   plans,
                                                   model)

    x, y = regression_trainer.construct_training_data(query_execution_df)

    num_epochs = 3
    training_history = regression_trainer.train(x, y, epochs=num_epochs,
                                                batch_size=len(y))

    self.assertLen(training_history.history.keys(),
                   1 + len(test_util.TEST_MODEL_CONFIG_2.metrics))
    self.assertIn("loss", training_history.history)
    self.assertIn("mean_absolute_error", training_history.history)
    self.assertLen(training_history.history["loss"], num_epochs)
    self.assertLen(training_history.history["mean_absolute_error"],
                   num_epochs)


if __name__ == "__main__":
  absltest.main()

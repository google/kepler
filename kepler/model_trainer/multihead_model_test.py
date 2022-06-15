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

"""Tests for multihead model."""

import numpy as np
import tensorflow as tf

from kepler.model_trainer import multihead_model
from kepler.model_trainer import test_util
from absl.testing import absltest
from absl.testing import parameterized


_MODEL_0 = multihead_model.MultiheadModel(
    test_util.TEST_METADATA_0, list(range(test_util.TEST_NUM_PLANS_0)),
    test_util.TEST_MODEL_CONFIG_0, test_util.TEST_PREPROCESSING_CONFIG_0)

_MODEL_1 = multihead_model.MultiheadModel(
    test_util.TEST_METADATA_0, list(range(test_util.TEST_NUM_PLANS_1)),
    test_util.TEST_MODEL_CONFIG_1, test_util.TEST_PREPROCESSING_CONFIG_0)


class MultiheadModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_NUM_PLANS_0),
      ("model1", _MODEL_1, test_util.TEST_NUM_PLANS_1)
  )
  def test_basic_training(self, model, num_plans):
    # Basic check that this model can train without raising errors.
    x = test_util.TEST_INPUT_PARAMS_0
    model.get_model().fit(x, np.zeros((4, num_plans)), epochs=1)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_NUM_PLANS_0),
      ("model1", _MODEL_1, test_util.TEST_NUM_PLANS_1)
  )
  def test_output_dimensions(self, model, num_plans):
    self.assertEqual(int(model.get_model().output.shape[1]), num_plans)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_NUM_PLANS_0),
      ("model1", _MODEL_1, test_util.TEST_NUM_PLANS_1)
  )
  def test_inference_dimensions(self, model, num_plans):
    self.assertEqual(
        model.get_model_outputs(test_util.TEST_INPUT_PARAMS_0).shape[1],
        num_plans)

  @parameterized.named_parameters(
      ("model0", _MODEL_0),
      ("model1", _MODEL_1)
  )
  def test_prediction_dimensions(self, model):
    self.assertTrue(test_util.test_model_prediction_shape(model))

  @parameterized.named_parameters(
      ("model0", _MODEL_0, tf.keras.activations.relu),
      ("model1", _MODEL_1, tf.keras.activations.tanh)
  )
  def test_activations(self, model, activation):
    for layer in model.get_model().layers:
      if "intermediate_dense_" in layer.name:
        self.assertEqual(layer.activation, activation)
      elif layer.name == "output_dense":
        self.assertEqual(layer.activation, tf.keras.activations.linear)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_MODEL_CONFIG_0,
       test_util.TEST_NUM_PLANS_0),
      ("model1", _MODEL_1, test_util.TEST_MODEL_CONFIG_1,
       test_util.TEST_NUM_PLANS_1)
  )
  def test_layer_widths(self, model, model_config, num_plans):
    for layer in model.get_model().layers:
      if "intermediate_dense_" in layer.name:
        idx = int(layer.name.split("_")[2])
        self.assertEqual(layer.units, model_config.layer_sizes[idx])
      elif layer.name == "output_dense":
        self.assertEqual(layer.units, num_plans)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_MODEL_CONFIG_0),
      ("model1", _MODEL_1, test_util.TEST_MODEL_CONFIG_1)
  )
  def test_dropout_rates(self, model, model_config):
    for layer in model.get_model().layers:
      if "dropout_" in layer.name:
        idx = int(layer.name.split("_")[1])
        self.assertEqual(layer.rate, model_config.dropout_rates[idx])

  @parameterized.named_parameters(
      ("model0", _MODEL_0, 1),
      ("model1", _MODEL_1, 2)
  )
  def test_num_layers(self, model, target_num_layers):
    self.assertLen([l for l in model.get_model().layers
                    if "intermediate_dense_" in l.name],
                   target_num_layers)


if __name__ == "__main__":
  absltest.main()

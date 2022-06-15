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

"""Tests for model_base."""

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from kepler.model_trainer import model_base
from kepler.model_trainer import test_util
from absl.testing import absltest
from absl.testing import parameterized


class ConcreteModelBase(model_base.ModelBase):
  """Make non-abstract class for testing."""
  _inputs: List[tf.keras.Input]  # pytype: disable=invalid-annotation  # typed-keras

  def __init__(self, metadata: Mapping[str, Any], plan_ids: List[int],
               model_config: Optional[model_base.ModelConfig],
               preprocessing_config: Sequence[Mapping[str, Any]]):
    super().__init__(metadata, plan_ids, model_config, preprocessing_config)

    self.preprocessing_layer = self._construct_preprocessing_layer()
    self._model = tf.keras.models.Model(
        inputs=self._inputs, outputs=self.preprocessing_layer)

  def get_model(self):
    return self._model

  def _get_model_predictions_helper(self, features):
    """Get predicted best plans."""
    return np.array([0] * len(features[0]))

  def get_model_outputs(self, features):
    """Perform forward inference."""
    raise NotImplementedError


class ModelBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # ModelBase does not use any model configs nor num_plans.
    self.base_model = ConcreteModelBase(test_util.TEST_METADATA_0, [], None,
                                        test_util.TEST_PREPROCESSING_CONFIG_0)
    self.model = self.base_model.get_model()

  def test_preprocessing_dimension(self):
    self.assertEqual(self.base_model.preprocessing_layer.dtype, tf.float32)
    self.assertEqual(int(self.base_model.preprocessing_layer.shape[1]), 141)

  def test_input_layers(self):
    # Inputs should all be dimension 1.
    for input_layer in self.base_model._inputs:
      self.assertEqual(int(input_layer.shape[1]), 1)

  @parameterized.named_parameters(
      ("param0", "embedding_preprocessing_param0", 11, 100),
      ("param1", "embedding_preprocessing_param1", 5, 10),
      ("param3", "embedding_preprocessing_param3", 33, 5)
  )
  def test_embedding_layer_dims(self, layer_name, input_dim, output_dim):
    # Input dimension for string embeddings is vocab size + num oov indices.
    self.assertIn(layer_name, [l.name for l in self.model.layers])
    layer = self.model.get_layer(layer_name)
    self.assertEqual(layer.input_dim, input_dim)
    self.assertEqual(layer.output_dim, output_dim)

  @parameterized.named_parameters(
      ("param4", "one_hot_preprocessing_param4", 21),
      ("param5", "one_hot_preprocessing_param5", 4),
  )
  def test_one_hot_layer_dims(self, layer_name, num_tokens):
    self.assertIn(layer_name, [l.name for l in self.model.layers])
    layer = self.model.get_layer(layer_name)
    self.assertEqual(layer.num_tokens, num_tokens)

  @parameterized.named_parameters(
      ("param0", "normalization_preprocessing_param2", 10, 5),)
  def test_normalization_layers(self, layer_name, mean, variance):
    # Tests that normalization layers have correct mean and variance.
    self.assertIn(layer_name, [l.name for l in self.model.layers])
    layer = self.model.get_layer(layer_name)
    self.assertEqual(layer.mean, mean)
    self.assertEqual(layer.variance, variance)

  @parameterized.named_parameters(
      ("case0", [0, 1], [0]),
      ("case1", [1, 2], [1]),
  )
  def test_get_model_predictions(self, plan_ids, expected_outputs):
    """Tests that we get the right plan ids."""
    model = ConcreteModelBase(test_util.TEST_METADATA_0, plan_ids, None,
                              test_util.TEST_PREPROCESSING_CONFIG_0)
    inputs = [np.zeros(1)]
    outputs = model.get_model_predictions(inputs)
    self.assertEqual(outputs, expected_outputs)


if __name__ == "__main__":
  absltest.main()

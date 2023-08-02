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

"""Tests for SNGP multihead model."""

import copy
from typing import Any, List

import numpy as np
import tensorflow as tf

from kepler.model_trainer import model_base
from kepler.model_trainer import sngp_multihead_model
from kepler.model_trainer import test_util
from kepler.model_trainer import trainer_util
from absl.testing import absltest
from absl.testing import parameterized


_MODEL_0 = sngp_multihead_model.SNGPMultiheadModel(
    test_util.TEST_METADATA_0, list(range(test_util.TEST_NUM_PLANS_0)),
    test_util.TEST_MODEL_CONFIG_0, test_util.TEST_PREPROCESSING_CONFIG_0)

_MODEL_1 = sngp_multihead_model.SNGPMultiheadModel(
    test_util.TEST_METADATA_0, list(range(test_util.TEST_NUM_PLANS_1)),
    test_util.TEST_MODEL_CONFIG_1, test_util.TEST_PREPROCESSING_CONFIG_0)


class SNGPMultiheadModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_NUM_PLANS_0),
      ("model1", _MODEL_1, test_util.TEST_NUM_PLANS_1)
  )
  def test_output_dimensions(self, model, num_plans):
    self.assertEqual(int(model.get_model().output[0].shape[1]), num_plans)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, 1),
      ("model1", _MODEL_1, 2)
  )
  def test_num_layers(self, model, target_num_layers):
    self.assertLen([l for l in model.get_model().layers
                    if "spectral_norm_" in l.name],
                   target_num_layers)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_MODEL_CONFIG_0),
      ("model1", _MODEL_1, test_util.TEST_MODEL_CONFIG_1)
  )
  def test_spectral_norm_multiplier(self, model, model_config):
    num_spectral_norm_layers = 0
    for layer in model.get_model().layers:
      if "spectral_norm_" in layer.name:
        self.assertEqual(layer.norm_multiplier,
                         model_config.spectral_norm_multiplier)
        num_spectral_norm_layers += 1
    self.assertLen(model_config.layer_sizes, num_spectral_norm_layers)

  @parameterized.named_parameters(
      ("model0", _MODEL_0, test_util.TEST_MODEL_CONFIG_0),
      ("model1", _MODEL_1, test_util.TEST_MODEL_CONFIG_1)
  )
  def test_num_gp_features(self, model, model_config):
    contains_output_gp_layer = False
    for layer in model.get_model().layers:
      if "output_gp_layer" == layer.name:
        self.assertEqual(layer.num_inducing,
                         model_config.num_gp_random_features)
        contains_output_gp_layer = True
    self.assertTrue(contains_output_gp_layer)


def _get_tflite_predictor(
    x: np.ndarray,
    model: tf.keras.Model,
    metadata: Any,
    plan_cover: List[int],
    confidence_threshold: float,
    train: bool,
) -> model_base.ModelPredictorBase:
  if train:
    # Set the 4th plan as optimal for the first 2 inputs and the 2nd plan as
    # optional for the latter 2 inputs.
    y = np.zeros((4, test_util.TEST_NUM_PLANS_0))
    y[0, 3] = 1
    y[1, 3] = 1
    y[2, 1] = 1
    y[3, 1] = 1

    model.fit(x, y, epochs=50)

  tflite_model = trainer_util.convert_to_tflite(model)
  return sngp_multihead_model.SNGPMultiheadModelTFLitePredictor(
      tflite_model_content=tflite_model,
      metadata=metadata,
      plan_cover=plan_cover,
      confidence_threshold=confidence_threshold,
  )


def _get_keras_model_predictor(
    x: np.ndarray,
    model: tf.keras.Model,
    metadata: Any,
    plan_cover: List[int],
    confidence_threshold: float,
    train: bool,
) -> model_base.ModelPredictorBase:
  if train:
    # Set the 4th plan as optimal for the first 2 inputs and the 2nd plan as
    # optional for the latter 2 inputs.
    y = np.zeros((4, test_util.TEST_NUM_PLANS_0))
    y[0, 3] = 1
    y[1, 3] = 1
    y[2, 1] = 1
    y[3, 1] = 1

    model.fit(x, y, epochs=50)

  return sngp_multihead_model.SNGPMultiheadModelPredictor(
      model=model,
      metadata=metadata,
      plan_cover=plan_cover,
      confidence_threshold=confidence_threshold,
  )


class SNGPMultiheadModelPredictorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._model = _MODEL_0.get_model()
    self._x_train = copy.deepcopy(test_util.TEST_INPUT_PARAMS_0)
    self._x_input = test_util.TEST_INPUT_PARAMS_0
    self._metadata = test_util.TEST_METADATA_0
    self._plan_cover = [10, 11, 12, 13, 14]
    trainer_util.apply_preprocessing(
        self._x_train, self._metadata["predicates"]
    )

  # The expected_predictions are based on the values set for y in
  # _get_tflite_model().
  @parameterized.named_parameters(
      dict(
          testcase_name="tflite low abstention",
          get_predictor_fn=_get_tflite_predictor,
          confidence_threshold=0.6,
          train=True,
          expected_predictions=[13, 13, 11, 11],
          batch_inference=False,
      ),
      dict(
          testcase_name="tflite low abstention batch",
          get_predictor_fn=_get_tflite_predictor,
          confidence_threshold=0.6,
          train=True,
          expected_predictions=[13, 13, 11, 11],
          batch_inference=True,
      ),
      dict(
          testcase_name="tflite high abstention",
          get_predictor_fn=_get_tflite_predictor,
          confidence_threshold=0.95,
          train=True,
          expected_predictions=[None, None, None, None],
          batch_inference=False,
      ),
      dict(
          testcase_name="keras model low abstention",
          get_predictor_fn=_get_keras_model_predictor,
          confidence_threshold=0.6,
          train=True,
          expected_predictions=[13, 13, 11, 11],
          batch_inference=False,
      ),
      dict(
          testcase_name="keras model high abstention",
          get_predictor_fn=_get_keras_model_predictor,
          confidence_threshold=0.95,
          train=True,
          expected_predictions=[None, None, None, None],
          batch_inference=False,
      ),
  )
  def test_predict(
      self, get_predictor_fn, confidence_threshold, train, expected_predictions,
      batch_inference
  ):
    predictor = get_predictor_fn(
        x=self._x_train,
        model=self._model,
        metadata=self._metadata,
        plan_cover=self._plan_cover,
        train=train,
        confidence_threshold=confidence_threshold,
    )

    for i, params in enumerate(zip(*self._x_input)):
      if batch_inference:
        params = [[p] for p in params]
      prediction, auxiliary = predictor.predict(params)
      self.assertEqual(prediction, expected_predictions[i])
      self.assertIn("confidences", auxiliary)
      self.assertLen(auxiliary["confidences"][0], test_util.TEST_NUM_PLANS_0)

  # The expected_predictions are based on the values set for y in
  # _get_tflite_model().
  @parameterized.named_parameters(
      dict(
          testcase_name="tflite model",
          get_predictor_fn=_get_tflite_predictor,
          confidence_threshold=0.2,
          train=False,
      ),
      dict(
          testcase_name="keras model",
          get_predictor_fn=_get_keras_model_predictor,
          confidence_threshold=0.2,
          train=False,
      ),
  )
  def test_illegal_calls(self, get_predictor_fn, confidence_threshold, train):
    predictor = get_predictor_fn(
        x=self._x_train,
        model=self._model,
        metadata=self._metadata,
        plan_cover=self._plan_cover,
        train=train,
        confidence_threshold=confidence_threshold,
    )

    self.assertRaisesRegex(
        ValueError,
        "Expected 7 parameter values and received 1 instead",
        predictor.predict,
        [1],
    )

    self.assertRaisesRegex(
        ValueError,
        "could not convert string to float",
        predictor.predict,
        ["a"] * 7,
    )

    short_metadata = {"predicates": [1]}
    self.assertRaisesRegex(
        ValueError,
        "Provided 1 predicates for a model",
        get_predictor_fn,
        self._x_train,
        self._model,
        short_metadata,
        self._plan_cover,
        train,
        confidence_threshold,
    )

    short_plan_cover = [1]
    self.assertRaisesRegex(
        ValueError,
        r"Provided plan cover size of 1",
        get_predictor_fn,
        self._x_train,
        self._model,
        self._metadata,
        short_plan_cover,
        train,
        confidence_threshold,
    )


if __name__ == "__main__":
  absltest.main()

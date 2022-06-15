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

"""Class for pairwise ranking model.

Applies standard pairwise learning to rank approach to ranking plans.
"""
from typing import Any, List, Mapping, Sequence

import numpy as np
import tensorflow as tf

from kepler.model_trainer import model_base

JSON = Any


class PairwiseRankingModel(model_base.ModelBase):
  """Pairwise ranking model for Kepler.

  This model attempts to compare pairs of plans by feeding in the inputs and
  each plan_id through a subnetwork outputting a single scalar. Then, the
  probability that one plan is better than the other is modeled as a sigmoid
  of the difference between their respective scores.
  """
  _inputs: List[tf.keras.Input]  # pytype: disable=invalid-annotation  # typed-keras

  def __init__(self, metadata: JSON, plan_ids: List[int],
               model_config: model_base.ModelConfig,
               preprocessing_config: Sequence[Mapping[str, Any]]):
    super().__init__(metadata, plan_ids, model_config, preprocessing_config)
    self._build_model()

  def _build_model(self):
    """Builds pairwise ranking model.

    Creates two models: ranker, which outputs a scalar score for a specified
    plan id, and model, which trains the ranker by taking two ranker inputs
    and training their sigmoid preference against the actual preference.
    """
    preprocessing_layer = self._construct_preprocessing_layer()

    # Create a plan_id input and map it to a one_hot tensor.
    plan_layer = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="plan_id")
    one_hot = tf.one_hot(plan_layer, self._num_plans)

    prev_layer = tf.keras.layers.Concatenate()([preprocessing_layer, one_hot])
    for i, (layer_size, dropout_rate) in enumerate(
        zip(self._model_config.layer_sizes, self._model_config.dropout_rates)):
      dense_layer = tf.keras.layers.Dense(
          layer_size,
          activation=self._model_config.activation,
          name=f"intermediate_dense_{i}")(
              prev_layer)
      prev_layer = tf.keras.layers.Dropout(
          dropout_rate, name=f"dropout_{i}")(
              dense_layer)

    output_layer = tf.keras.layers.Dense(1, name="output_dense")(prev_layer)
    self._ranker = tf.keras.models.Model(
        inputs=self._inputs + [plan_layer], outputs=output_layer)

    # Construct pairwise ranking model.
    input_layer_a = [
        self._input_layer(p["data_type"], "param%d_a" % i)
        for i, p in enumerate(self._predicate_metadata)
    ] + [tf.keras.layers.Input(shape=(), dtype=tf.int32, name="plan_id_a")]
    input_layer_b = [
        self._input_layer(p["data_type"], "param%d_b" % i)
        for i, p in enumerate(self._predicate_metadata)
    ] + [tf.keras.layers.Input(shape=(), dtype=tf.int32, name="plan_id_b")]

    score_a = self._ranker(input_layer_a)
    score_b = self._ranker(input_layer_b)
    probability = tf.math.sigmoid(tf.math.subtract(score_a, score_b))
    self._model = tf.keras.models.Model(
        inputs=[input_layer_a, input_layer_b], outputs=probability)

    learning_rate = self._model_config.learning_rate
    self._model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=self._model_config.loss,
        metrics=self._model_config.metrics)

  def get_model(self) -> tf.keras.Model:  # pytype: disable=invalid-annotation  # typed-keras
    """Get entire pairwise ranking model."""
    return self._model

  def _get_model_predictions_helper(self,
                                    params: List[np.ndarray]) -> np.ndarray:
    """Get model best plan predictions for a batch of parameters.

    To do this, we run model inference for all plan ids and all parameters.

    Args:
      params: Batch of parameter values to input to the model. Each entry in
        params contains all values for a single input feature.

    Returns:
      All predicted best plan indices.
    """
    num_params = len(params[0])
    # Repeat each param self._num_plans times and then concatenate the
    # plan_id to the each copy of the param.
    input_features = ([np.repeat(x, self._num_plans) for x in params] +
                      [np.tile(np.arange(self._num_plans), (num_params,))])
    # Return the best plan for each parameter.
    plan_outputs = self.get_model_outputs(input_features).reshape(
        -1, self._num_plans)
    return np.argmax(plan_outputs, axis=1)

  def get_model_outputs(self, params: List[np.ndarray]) -> np.ndarray:
    """Perform model inference on a batch of parameters.

    Args:
      params: Batch of parameter values to input to the model. Each entry in
        params contains all values for a single input feature. The last entry in
        params will correspond to the plan_ids.

    Returns:
      All model outputs.
    """
    input_features = self._cast_inputs_np_type(params[:-1]) + [params[-1]]
    return self._ranker.predict_on_batch(input_features)

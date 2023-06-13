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

"""Class for multihead model, for which there is an output head for each plan.

In general, each output value indicates to the how good the corresponding plan
is estimated to be. The predicted plan will be the plan corresponding to the
head with the highest value.

This model can be trained in a variety of ways. For example, in the
classification setting, the outputs will correspond to the probability that
that plan is the optimal plan for the inputs. One can also define
latency-sensitive training losses for this model.
"""

from typing import Any, List, Mapping, Sequence

import tensorflow as tf

from kepler.model_trainer import model_base

JSON = Any


class MultiheadModel(model_base.ModelBase):
  """Model for multihead neural network model.

  The model configurations for this model are:
    - layer sizes (and number of layers)
    - dropout rates for each layer
    - optimizer learning rate
    - activation function
    - loss function. For example, with no softmax activation, we need to adjust
        the loss appropriately (standard categorical_crossentropy won't work).
  """
  _inputs: List[tf.keras.Input]  # pytype: disable=invalid-annotation  # typed-keras

  def __init__(self, metadata: JSON, plan_ids: List[int],
               model_config: model_base.ModelConfig,
               preprocessing_config: Sequence[Mapping[str, Any]]):
    self._initialize_base(metadata, plan_ids, model_config,
                          preprocessing_config)
    self._build_model()

  def _initialize_base(self, metadata: JSON, plan_ids: List[int],
                       model_config: model_base.ModelConfig,
                       preprocessing_config: Sequence[Mapping[str, Any]]):
    super().__init__(metadata, plan_ids, model_config, preprocessing_config)

  # LINT.IfChange
  def _build_model(self) -> None:
    """Constructs model via Keras Functional API."""
    prev_layer = self._construct_preprocessing_layer()
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

    result = tf.keras.layers.Dense(
        self._num_plans, name="output_dense")(
            prev_layer)
    model = tf.keras.models.Model(inputs=self._inputs, outputs=result)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=self._model_config.learning_rate),
        loss=self._model_config.loss,
        metrics=self._model_config.metrics)

    self._model = model
  # LINT.ThenChange(//depot/google3/research/sir/kepler/model_trainer/sngp_multihead_model.py)

  def get_model(self) -> tf.keras.Model:  # pytype: disable=invalid-annotation  # typed-keras
    return self._model

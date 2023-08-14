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

"""Common utils/constants for tests."""

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from kepler.model_trainer import loss_functions
from kepler.model_trainer import model_base


class ConcreteModelBase(model_base.ModelBase):
  """Make non-abstract class for testing."""

  _inputs: List[tf.keras.Input]  # pytype: disable=invalid-annotation  # typed-keras

  def __init__(
      self,
      metadata: Mapping[str, Any],
      plan_ids: List[int],
      model_config: Optional[model_base.ModelConfig],
      preprocessing_config: Sequence[Mapping[str, Any]],
  ):
    super().__init__(metadata, plan_ids, model_config, preprocessing_config)

    self.preprocessing_layer = self._construct_preprocessing_layer()
    self._model = tf.keras.models.Model(
        inputs=self._inputs, outputs=self.preprocessing_layer
    )

  def get_model(self) -> tf.keras.Model:
    return self._model


# Only contains a subset of predicates metadata by design. Actual metadata
# has other keys, which are unused and thus omitted for the sake of brevity.
TEST_METADATA_0 = {
    "predicates": [
        {
            "alias": "s",
            "column": "name",
            "operator": "=",
            "data_type": "text",
            "distinct_values": ["a", "b", "c", "d", "e", "f"],
        },
        {
            "alias": "t",
            "column": "name",
            "operator": "=",
            "data_type": "text",
            "distinct_values": ["aa", "bb", "cc", "dd", "ee"],
        },
        {
            "alias": "q",
            "column": "score",
            "operator": ">",
            "data_type": "float",
        },
        {
            "alias": "q",
            "column": "view_count",
            "operator": "<",
            "data_type": "int",
            "min": -5,
            "max": 27,
        },
        {
            "alias": "q",
            "column": "upvotes",
            "operator": "<",
            "data_type": "int",
            "min": 0,
            "max": 20,
        },
        {
            "alias": "b",
            "column": "name",
            "operator": "=",
            "data_type": "text",
            "distinct_values": ["a", "b", "c"],
        },
        {
            "alias": "b",
            "column": "name",
            "operator": "=",
            "data_type": "float",
            "preprocess_type": "to_timestamp",
        },
    ]
}

TEST_PREPROCESSING_CONFIG_0 = [
    {"type": "embedding", "output_dim": 100, "num_oov_indices": 5},
    {"type": "embedding", "output_dim": 10},
    {"type": "std_normalization", "mean": 10, "variance": 5},
    {"type": "embedding", "output_dim": 5},
    {"type": "one_hot"},
    {"type": "one_hot", "num_oov_indices": 1},
    {"type": "std_normalization", "mean": 1453276800, "variance": 1e10},
]

# Corresponds to data_management test_query 0.
TEST_PREPROCESSING_CONFIG_1 = [{
    "type": "embedding",
    "output_dim": 100,
    "num_oov_indices": 5
}, {
    "type": "embedding",
    "output_dim": 100,
    "num_oov_indices": 1,
}, {
    "type": "embedding",
    "output_dim": 10
}, {
    "type": "std_normalization",
    "mean": 0,
    "variance": 1
}]

TEST_INPUT_PARAMS_0 = [
    np.array(["a", "c", "e", "z"]),
    np.array(["aa", "bb", "cc", "ee"]),
    np.ones(4),
    np.arange(4),
    np.arange(4),
    np.array(["a"] * 4),
    np.array(["2016-01-20", "2016-01-20", "2016-01-20", "2016-01-20"]),
]

# Note: We would not want to actually train the model using the default BCE
# loss, since we assume linear output activation. Instead, in the classification
# setting one should use BCE loss with from_logits set to True.
TEST_MODEL_CONFIG_0 = model_base.ModelConfig(
    [64],
    [0.1],
    1e-3,
    "relu",
    tf.keras.losses.BinaryCrossentropy(from_logits=True),
    [],
)
TEST_NUM_PLANS_0 = 5

TEST_MODEL_CONFIG_1 = model_base.ModelConfig(
    [64, 32], [0.25, 0.2], 1e-3, "tanh", "binary_crossentropy", ["accuracy"]
)
TEST_NUM_PLANS_1 = 10

TEST_MODEL_CONFIG_2 = model_base.ModelConfig(
    [64], [0.1], 1e-3, "relu", loss_functions.mse_loss,
    [tf.keras.metrics.MeanAbsoluteError()])

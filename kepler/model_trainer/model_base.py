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

"""Base functionality for Kepler models.

Contains:
  - ModelConfig class for specifying model architectures and hyperparams.
  - ModelBase class as an abstract base class for any Kepler model.
"""
import abc
import dataclasses
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from kepler.model_trainer import trainer_util

JSON = Any


@dataclasses.dataclass(frozen=True)
class ModelConfig():
  """Stores model architecture/hyperparameters.

  TODO(b/217974851): Options for optimizer, weight/batch norm, etc.
  """
  layer_sizes: List[int]  # All intermediate hidden layer sizes.
  dropout_rates: List[float]  # Dropout rate for each layer.
  learning_rate: float
  activation: str
  loss: Any  # Loss can be custom function or string.
  metrics: List[Any]


def _apply_preprocessing_layer(
    name: str,
    param: Mapping[str, Any],
    layer: tf.keras.Input,  # pytype: disable=invalid-annotation  # typed-keras
    preprocessing_info: Mapping[str, Any]
) -> Optional[tf.keras.layers.Layer]:  # pytype: disable=invalid-annotation  # typed-keras
  """Generates the preprocessing layer for a given input parameter.

  We currently support the following types of preprocessing:
    - Normalization to mean 0, variance 1.
    - Embedding of integer types. This requires specifying min/max of
      the integer values, since we can only embed non-negative values.
    - String embeddings.

  Args:
    name: Suffix for names of produced layers.
    param: Contains info about the parameter; a single entry in
      self.predicate_metadata.
    layer: Keras input layer for this parameter.
    preprocessing_info: Contains info about how to preprocess this parameter; a
      single entry in self.preprocessing_config.

  Returns:
    The preprocessing layer for the parameter.

  Raises:
    ValueError: If the parameter type or preprocessing type is invalid.
  """
  data_type = param["data_type"]
  preprocessing_type = preprocessing_info["type"]
  if data_type == "float" and preprocessing_type == "std_normalization":
    norm = tf.keras.layers.experimental.preprocessing.Normalization(
        mean=preprocessing_info["mean"],
        variance=preprocessing_info["variance"],
        name=f"normalization_{name}")
    return norm(layer)
  if data_type == "int":
    # Shift layer to be non-negative.
    shifted_layer = layer - tf.constant(param["min"], dtype=tf.int64)
    if preprocessing_type == "embedding":
      embedding = tf.keras.layers.Embedding(
          input_dim=param["max"] - param["min"] + 1,
          output_dim=preprocessing_info["output_dim"],
          name=f"embedding_{name}")(
              shifted_layer)
      return tf.keras.layers.Flatten()(embedding)
    elif preprocessing_type == "one_hot":
      onehot_layer = tf.keras.layers.CategoryEncoding(
          num_tokens=param["max"] - param["min"] + 1,
          output_mode="one_hot",
          name=f"one_hot_{name}")(
              shifted_layer)
      return tf.keras.layers.Flatten()(onehot_layer)
  if data_type == "text":
    vocabulary = param["distinct_values"]
    num_oov_indices = preprocessing_info.get("num_oov_indices", 0)
    lookup_layer = tf.keras.layers.StringLookup(
        num_oov_indices=num_oov_indices,
        vocabulary=vocabulary,
        name=f"lookup_{name}")(
            layer)
    if preprocessing_type == "embedding":
      embedding_layer = tf.keras.layers.Embedding(
          input_dim=len(vocabulary) + num_oov_indices,
          output_dim=preprocessing_info["output_dim"],
          name=f"embedding_{name}")(
              lookup_layer)
      return tf.keras.layers.Flatten()(embedding_layer)
    elif preprocessing_type == "one_hot":
      onehot_layer = tf.keras.layers.CategoryEncoding(
          num_tokens=len(vocabulary) + num_oov_indices,
          output_mode="one_hot",
          name=f"one_hot_{name}")(
              lookup_layer)
      return tf.keras.layers.Flatten()(onehot_layer)

  raise ValueError(f"Unsupported preprocessing: parameter type: {data_type}"
                   f" preprocessing type: {preprocessing_type}")


class ModelBase(metaclass=abc.ABCMeta):
  """Abstract base for classes that implement a Kepler model.

  A Kepler model is distinguished by the parameters of its corresponding
  query template. This base class contains logic to handle input parameters
  of varying types (str, int, float), as well as multiple ways to preprocess
  each type, e.g. normalization and embedding.

  This class also contains abstract methods for performing model training,
  inference, prediction, etc.
  """

  def __init__(self, metadata: JSON,
               plan_ids: List[int],
               model_config: Optional[ModelConfig],
               preprocessing_config: Sequence[Mapping[str, Any]]):
    """Initializes base model.

    Args:
      metadata: Metadata for entire query (i.e. under query_id key). The
        predicates value is structured as a list, with each entry corresponding
        to a param. Each entry in this list is a dict containing data_type (int,
        float, text), and associated info, e.g. min/max ranges for ints,
        distinct values for embedded parameters, etc.
      plan_ids: List of plan ids to predict.
      model_config: Model configuration data.
      preprocessing_config: Preprocessing config for each parameter, structured
        as a list corresponding to the elements of metadata['predicates'].
        For example:
          [{"type": "embedding"},
           {"type": "std_normalization", "mean": 0., "variance": 1.}]

    Raises:
      ValueError: If predicates metadata and preprocessing config don't have
        the same length.
    """
    if len(metadata["predicates"]) != len(preprocessing_config):
      raise ValueError("Predicates metadata and preprocessing config have "
                       "mismatched lengths! %d != %d" %
                       (len(metadata["predicates"]),
                        len(preprocessing_config)))

    self._predicate_metadata = metadata["predicates"]
    self._num_plans = len(plan_ids)
    self._model_index_to_plan_id = {i: plan_id for i, plan_id in
                                    enumerate(plan_ids)}

    self._model_config = model_config
    self._preprocessing_config = preprocessing_config

  def _cast_inputs_np_type(self, params: List[np.ndarray]) -> List[np.ndarray]:
    """Casts np inputs to proper types to avoid tf 2 casting errors.

    This seems to only be necessary in tf 2 eager execution.

    Args:
      params: Inputs to model inference.

    Returns:
      Casted version of inputs.
    """
    return [
        feature.astype(trainer_util.get_np_type(param_data["data_type"]))
        for feature, param_data in zip(params, self._predicate_metadata)
    ]

  def _construct_preprocessing_layer(self) -> tf.keras.layers.Layer:  # pytype: disable=invalid-annotation  # typed-keras
    """Constructs input layer and preprocessing layer.

    Returns:
      Concatenated preprocessing layer for the model.
    """
    self._inputs = [
        self._input_layer(p["data_type"], f"param{i}")
        for i, p in enumerate(self._predicate_metadata)
    ]

    to_concatenate = []
    for i in range(len(self._predicate_metadata)):
      to_concatenate.append(
          _apply_preprocessing_layer(f"preprocessing_param{i}",
                                     self._predicate_metadata[i],
                                     self._inputs[i],
                                     self._preprocessing_config[i]))
    return tf.keras.layers.Concatenate()(to_concatenate)

  def _input_layer(
      self,
      data_type: str,
      name: Optional[str] = None
  ) -> tf.keras.Input:  # pytype: disable=invalid-annotation  # typed-keras
    """Return a 1d input layer with the appropriate data type."""
    return tf.keras.layers.Input(
        shape=(1), dtype=trainer_util.get_tf_type(data_type), name=name)

  @abc.abstractmethod
  def get_model(self) -> tf.keras.Model:  # pytype: disable=invalid-annotation  # typed-keras
    """Get keras model."""
    raise NotImplementedError

  @abc.abstractmethod
  def _get_model_predictions_helper(
      self, params: List[np.ndarray]) -> np.ndarray:
    """Get predicted best plans by model index."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_outputs(self, params: List[np.ndarray]) -> np.ndarray:
    """Performs forward inference."""
    raise NotImplementedError

  def get_model_predictions(
      self, params: List[np.ndarray]) -> np.ndarray:
    """Get plan ids corresponding to predicted best plans."""
    return np.array(list(map(lambda x: self._model_index_to_plan_id[x],
                             self._get_model_predictions_helper(params))))

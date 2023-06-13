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

"""Class for SNGP multihead model.

SNGP (Spectral-normalized Neural Gaussian Processes) improves uncertainty
quantification in neural networks by introducing 1. distance awareness via
spectral normalization, and 2. calibrated uncertainty outputs via a Gaussian
processs output layer. (https://arxiv.org/abs/2006.10108)

https://www.tensorflow.org/tutorials/understanding/sngp
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from kepler.model_trainer import model_base
from kepler.model_trainer import multihead_model
from kepler.model_trainer import trainer_util

JSON = Any


class ResetCovarianceCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch: int, logs: Any = None):
    """Resets covariance matrix at the beginning of the epoch."""
    if epoch > 0:
      self.model.classifier.reset_covariance_matrix()


class SNGPModel(tf.keras.models.Model):
  """Keras model with ResetCovarianceCallback."""

  def __init__(self, classifier: tfm.nlp.layers.RandomFeatureGaussianProcess,
               **kwargs):
    super().__init__(**kwargs)
    self.classifier = classifier

  def fit(self, *args: Any, **kwargs: Any) -> tf.keras.callbacks.History:
    """Adds ResetCovarianceCallback to model callbacks."""
    kwargs["callbacks"] = list(kwargs.get("callbacks", []))
    kwargs["callbacks"].append(ResetCovarianceCallback())

    return super().fit(*args, **kwargs)


class SNGPMultiheadModel(multihead_model.MultiheadModel):
  """Spectral-normalized Neural Gaussian Process model.

  This model is based on the multihead neural net model and introduces the
  following hyperparameters:
    - spectral_norm_multiplier: Bounds the maximum eigenvalue of each dense
      weight matrix.
    - num_gp_random_features: How many random features to use for the output
      GP layer.
  """
  _inputs: List[tf.keras.Input]  # pytype: disable=invalid-annotation  # typed-keras

  def __init__(self, metadata: JSON, plan_ids: List[int],
               model_config: model_base.ModelConfig,
               preprocessing_config: Sequence[Mapping[str, Any]]):
    self._initialize_base(metadata, plan_ids, model_config,
                          preprocessing_config)
    self._build_model()

  def _build_model(self) -> None:
    """Constructs model via Keras Functional API."""
    prev_layer = self._construct_preprocessing_layer()
    for i, (layer_size, dropout_rate) in enumerate(
        zip(self._model_config.layer_sizes, self._model_config.dropout_rates)):
      dense_layer = tf.keras.layers.Dense(
          layer_size,
          activation=self._model_config.activation)
      dense_layer = tfm.nlp.layers.SpectralNormalization(
          dense_layer,
          iteration=1,
          norm_multiplier=self._model_config.spectral_norm_multiplier,
          name=f"spectral_norm_{i}")
      prev_layer = tf.keras.layers.Dropout(
          dropout_rate, name=f"dropout_{i}")(
              dense_layer(prev_layer))

    # Hardcoded hyperparameters are same as those in SNGP tutorial.
    classifier = tfm.nlp.layers.RandomFeatureGaussianProcess(
        self._num_plans,
        num_inducing=self._model_config.num_gp_random_features,
        normalize_input=False,
        scale_random_features=True,
        gp_cov_momentum=-1,
        name="output_gp_layer")
    logits, covariance_matrix = classifier(prev_layer)

    model = SNGPModel(
        classifier=classifier, inputs=self._inputs,
        outputs=[logits, covariance_matrix])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=self._model_config.learning_rate),
        loss=[self._model_config.loss, None],
        metrics=self._model_config.metrics)

    self._model = model


def _sngp_prediction_helper(
    logits: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_param: float,
    plan_cover: List[int],
    confidence_threshold: float,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
  """Computes confidences and chooses plan based on confidence_threshold."""
  logits_adjusted = tfm.nlp.layers.gaussian_process.mean_field_logits(
      logits=logits,
      covariance_matrix=covariance_matrix,
      mean_field_factor=lambda_param,
  )
  # Compute binary softmax since we're predicting whether each plan is
  # near-optimal.
  confidences = 1 / (1 + np.exp(-1 * logits_adjusted))

  auxiliary = {"confidences": confidences}
  plan_cover_predicted_indices = np.argmax(confidences, axis=1)
  plan_predictions = np.take(plan_cover, plan_cover_predicted_indices).astype(
      object
  )

  # Compare the confidence of the most confident prediction against a threshold.
  plan_predictions[np.max(confidences, axis=1) < confidence_threshold] = None

  return plan_predictions, auxiliary


class SNGPMultiheadModelPredictor(model_base.ModelPredictorBase):
  """Predictor for sngp-based multihead model."""

  def __init__(
      self,
      model: tf.keras.Model,
      metadata: JSON,
      plan_cover: List[int],
      confidence_threshold: float,
  ):
    """Set up a predictor using a keras model.

    Args:
      model: The keras model for prediction.
      metadata: Metadata for entire query (i.e. under query_id key). The
        predicates value is structured as a list, with each entry corresponding
        to a param. Each entry in this list is a dict containing data_type (int,
        float, text), and associated info, e.g. min/max ranges for ints,
        distinct values for embedded parameters, etc.
      plan_cover: The plan cover with which the model was trained. The plan
        cover maps the model predicted indices to query plan ids.
      confidence_threshold: The confidence threshold below which the model
        should abstain from predicting.

    Raises:
      ValueError: If the metadata predicates count doesn't match the number
        of model inputs or if the plan_cover size doesn't match the number of
        output heads.
    """
    self._model = model
    self._predicates_metadata = metadata["predicates"]
    self._plan_cover = plan_cover
    self._confidence_threshold = confidence_threshold

    if len(self._model.inputs) != len(self._predicates_metadata):
      raise ValueError(
          f"Provided {len(self._predicates_metadata)} predicates for a model"
          f" with {len(self._model.inputs)} inputs."
      )

    # The model has two outputs. The first is the logits per candidate plan,
    # which has shape (None, number of candidate plans).
    model_output_heads = self._model.outputs[0].shape[1]
    if model_output_heads != len(self._plan_cover):
      raise ValueError(
          f"Provided plan cover size of {len(self._plan_cover)} does not match"
          f" the model output heads count of {model_output_heads}."
      )

  def predict(
      self, params: List[Any], lambda_param: float = np.pi / 8
  ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Returns the predicted plan id from the model.

    Args:
      params: Inputs to model inference.
      lambda_param: Controls temperature of mean-field GP posterior
        approximation.

    Returns:
      A tuple containing:
        1. Array of plan ids predicted to give the best latency by the model, or
           None if the model abstains from making a prediction due to low
           confidence.
        2. An optional dictionary containing predictor-specific auxiliary
           values.

    Raises:
      ValueError: If the wrong number of params are provided or if the types of
        the provided params are incorrect.
    """
    if len(params) != len(self._model.inputs):
      raise ValueError(
          f"Expected {len(self._model.inputs)} parameter values and received"
          f" {len(params)} instead"
      )

    model_inputs = []
    for param, predicate in zip(params, self._predicates_metadata):
      model_inputs.append(
          model_base.prepare_input_param(
              param=param, predicate=predicate, tflite_mode=False
          )
      )

    logits, covariance_matrix = self._model.predict_on_batch(model_inputs)

    return _sngp_prediction_helper(
        logits=logits,
        covariance_matrix=covariance_matrix,
        lambda_param=lambda_param,
        plan_cover=self._plan_cover,
        confidence_threshold=self._confidence_threshold,
    )


class SNGPMultiheadModelTFLitePredictor(model_base.ModelPredictorBase):
  """Predictor for tflite version of sngp-based multihead model."""

  def __init__(
      self,
      tflite_model_content: bytes,
      metadata: JSON,
      plan_cover: List[int],
      confidence_threshold: float,
  ):
    """Set up a predictor using a tflite model.

    Args:
      tflite_model_content: The loaded model bytes from a model in the tflite
        format.
      metadata: Metadata for entire query (i.e. under query_id key). The
        predicates value is structured as a list, with each entry corresponding
        to a param. Each entry in this list is a dict containing data_type (int,
        float, text), and associated info, e.g. min/max ranges for ints,
        distinct values for embedded parameters, etc
      plan_cover: The plan cover with which the model was trained. The plan
        cover maps the model predicted indices to query plan ids.
      confidence_threshold: The confidence threshold below which the model
        should abstain from predicting.

    Raises:
      ValueError: If the metadata predicates count doesn't match the number
        of model inputs or if the plan_cover size doesn't match the number of
        output heads.
    """
    self._interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    self._interpreter.allocate_tensors()
    self._predicates_metadata = metadata["predicates"]
    self._plan_cover = plan_cover

    input_details = self._interpreter.get_input_details()

    if len(input_details) != len(self._predicates_metadata):
      raise ValueError(
          f"Provided {len(self._predicates_metadata)} predicates for a model"
          f" with {len(input_details)} inputs."
      )

    # The model has two outputs. The first is the logits per candidate plan,
    # which has shape (1, number of candidate plans).
    model_output_heads = self._interpreter.get_output_details()[0]["shape"][1]
    if model_output_heads != len(self._plan_cover):
      raise ValueError(
          f"Provided plan cover size of {len(self._plan_cover)} does not match"
          f" the model output heads count of {model_output_heads}."
      )

    self._parameter_index_map = {}
    for index, input_detail in enumerate(input_details):
      param = trainer_util.extract_tflite_parameter_index(input_detail)
      self._parameter_index_map[param] = index

    output_details = self._interpreter.get_output_details()
    self._logits_key = output_details[0]["index"]
    self._covariance_matrix_key = output_details[1]["index"]

    self._confidence_threshold = confidence_threshold

  def predict(
      self, params: List[Any], lambda_param: float = np.pi / 8
  ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Returns the predicted plan id from the model.

    Args:
      params: Inputs to model inference.
      lambda_param: Controls temperature of mean-field GP posterior
        approximation.

    Returns:
      A tuple containing:
        1. Array of plan ids predicted to give the best latency by the model,
           or None if the model abstains from making a prediction due to low
           confidence.
        2. An optional dictionary containing predictor-specific auxiliary
           values.

    Raises:
      ValueError: If the wrong number of params are provided or if the types of
        the provided params are incorrect.
    """
    if len(params) != len(self._predicates_metadata):
      raise ValueError(
          f"Expected {len(self._predicates_metadata)} parameter values and"
          f" received {len(params)} instead"
      )

    for index, (param, predicate) in enumerate(
        zip(params, self._predicates_metadata)
    ):
      tflite_param_index = self._parameter_index_map[index]
      self._interpreter.set_tensor(
          tflite_param_index,
          model_base.prepare_input_param(
              param=param, predicate=predicate, tflite_mode=True
          ),
      )

    self._interpreter.invoke()

    logits = self._interpreter.get_tensor(self._logits_key)
    covariance_matrix = self._interpreter.get_tensor(
        self._covariance_matrix_key
    )

    return _sngp_prediction_helper(
        logits=logits,
        covariance_matrix=covariance_matrix,
        lambda_param=lambda_param,
        plan_cover=self._plan_cover,
        confidence_threshold=self._confidence_threshold,
    )

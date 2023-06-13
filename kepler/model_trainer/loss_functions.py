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

"""Module for Kepler loss functions."""

import tensorflow as tf


def mse_loss(y_true: tf.Tensor,
             y_pred: tf.Tensor) -> tf.Tensor:
  """Computes mean squared error.

  Attempts to fit to the negative of the target to be compatible with
  argmax in MultiheadModel. The true target values are assumed to
  represent costs, so that the model learns to predict the negative cost.

  Args:
    y_true: True target costs (e.g. latencies).
    y_pred: Model predictions.

  Returns:
    Scalar tensor corresponding to the loss.
  """
  return tf.math.reduce_mean(tf.square(y_true + y_pred))


def log_mse_loss(y_true: tf.Tensor,
                 y_pred: tf.Tensor) -> tf.Tensor:
  """Computes mean squared error against log of true values.

  Like mse_loss, attempts to fit the negative cost. This loss uses the
  log of the cost for better stability.

  Args:
    y_true: True target costs (e.g. latencies).
    y_pred: Model predictions.

  Returns:
    Scalar tensor corresponding to the loss.
  """
  return tf.math.reduce_mean(tf.square(tf.math.log(y_true) + y_pred))

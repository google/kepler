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

"""Tests for loss_functions."""

import tensorflow as tf

from kepler.model_trainer import loss_functions
from absl.testing import absltest
from absl.testing import parameterized


class LossFunctionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("expected_zero", tf.constant([1., 2., 2.]), tf.constant([-1., -2., -2.]),
       tf.constant([0.])),
      ("batch_expected_zero",
       tf.constant([[1., 2., 2.], [2., 4., 4.], [3., 5., 5.]]),
       tf.constant([[-1., -2., -2.], [-2., -4., -4.], [-3., -5., -5.]]),
       tf.constant([0.])),
      ("same_nonzero", tf.constant([1., 2., 2.]), tf.constant([1., 2., 2.]),
       tf.constant([12.]))
  )
  def test_mse_loss(self, y_true, y_pred, expected_loss):
    loss = loss_functions.mse_loss(y_true, y_pred)
    self.assertTrue(tf.math.equal(loss, expected_loss))

  @parameterized.named_parameters(
      ("expected_zero", tf.constant([1., 1., 1.]),
       tf.constant([0., 0., 0.]),
       tf.constant([0.])),
      ("batch_expected_zero",
       tf.math.exp(tf.constant([[1., 2., 2.], [2., 4., 4.], [3., 5., 5.]])),
       tf.constant([[-1., -2., -2.], [-2., -4., -4.], [-3., -5., -5.]]),
       tf.constant([0.])),
      ("same_nonzero",
       tf.math.exp(tf.constant([1., 2., 2.])), tf.constant([1., 2., 2.]),
       tf.constant([12.]))
  )
  def test_log_mse_loss(self, y_true, y_pred, expected_loss):
    loss = loss_functions.log_mse_loss(y_true, y_pred)
    self.assertTrue(tf.math.equal(loss, expected_loss))


if __name__ == "__main__":
  absltest.main()

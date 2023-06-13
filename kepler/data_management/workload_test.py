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

"""Tests for Workload."""
import copy
from typing import List

from kepler.data_management import test_util
from kepler.data_management import workload
from absl.testing import absltest
from absl.testing import parameterized


class WorkloadGenerateTest(parameterized.TestCase):

  def test_kepler_plan_discoverer(self):
    plans = workload.KeplerPlanDiscoverer(
        query_execution_data=test_util.QUERY_EXECUTION_DATA)
    self.assertEqual(plans.plan_ids, [0, 1, 2, 3])

    plans = workload.KeplerPlanDiscoverer(
        query_execution_metadata=test_util.QUERY_EXECUTION_METADATA)
    self.assertEqual(plans.plan_ids, [0, 1, 2])

  def test_kepler_plan_discoverer_illegal_init_calls(self):
    self.assertRaisesRegex(
        ValueError,
        "Exactly one of query_execution_data and query_execution_metadata must",
        workload.KeplerPlanDiscoverer)

    self.assertRaisesRegex(
        ValueError,
        "Exactly one of query_execution_data and query_execution_metadata must",
        workload.KeplerPlanDiscoverer, test_util.QUERY_EXECUTION_DATA,
        test_util.QUERY_EXECUTION_METADATA)

  def test_workload_generator_properties(self):
    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)
    self.assertEqual(workload_generator.parameter_count, 4)
    self.assertLen(test_util.PARAMETERS_POOL,
                   workload_generator.workload_pool_size)

  def _verify_parameters_from_pool(
      self, query_log: List[workload.QueryInstance]) -> None:
    for query_instance in query_log:
      self.assertEqual(query_instance.execution_frequency, 1)
      self.assertIn(query_instance.parameters, test_util.PARAMETERS_POOL)

  @parameterized.named_parameters(
      dict(testcase_name="small", n=1), dict(testcase_name="medium", n=3),
      dict(testcase_name="all", n=4))
  def test_workload_generator_random_sample(self, n):
    """Verifies workloads generated as random samples.

    Each sample must be the right size, selected without replacement, and only
    contain members from the known population.

    Args:
      n: The number of parameters to sample.
    """

    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)

    random_workload = workload_generator.random_sample(n)
    self.assertEqual(random_workload.query_id, test_util.TEST_QUERY_ID)
    self.assertLen(random_workload.query_log, n)

    parameters_set = set(
        [tuple(entry.parameters) for entry in random_workload.query_log])
    self.assertLen(parameters_set, len(random_workload.query_log))

    self._verify_parameters_from_pool(random_workload.query_log)

  def test_shuffle(self):
    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)
    workload_all = workload_generator.all()
    self.assertLen(workload_all.query_log, 5)
    self.assertEqual(workload_all.query_id, test_util.TEST_QUERY_ID)
    self._verify_parameters_from_pool(workload_all.query_log)

    # Shuffling changes the instance order.
    current_query_log = copy.copy(workload_all.query_log)
    workload.shuffle(workload_all, seed=test_util.TEST_SEED)
    self.assertLen(workload_all.query_log, 5)
    self.assertEqual(workload_all.query_id, test_util.TEST_QUERY_ID)
    self._verify_parameters_from_pool(workload_all.query_log)

    self.assertNotEqual(current_query_log, workload_all.query_log)

    # Shuffling again with the same seed provides the same instance order.
    workload_all_again = workload_generator.all()
    workload.shuffle(workload_all_again, seed=test_util.TEST_SEED)
    self.assertLen(workload_all_again.query_log, 5)
    self.assertEqual(workload_all_again.query_id, test_util.TEST_QUERY_ID)
    self._verify_parameters_from_pool(workload_all_again.query_log)

    self.assertEqual(workload_all_again.query_log, workload_all.query_log)

    # Shuffling with a new seed provides a new instance order.
    workload_all_new_seed = workload_generator.all()
    workload.shuffle(workload_all_new_seed, seed=test_util.TEST_SEED + 1)
    self.assertLen(workload_all_new_seed.query_log, 5)
    self.assertEqual(workload_all_new_seed.query_id, test_util.TEST_QUERY_ID)
    self._verify_parameters_from_pool(workload_all_new_seed.query_log)

    self.assertNotEqual(workload_all_new_seed.query_log, workload_all.query_log)

  def test_split_illegal_calls(self):
    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)
    workload_all = workload_generator.all()

    self.assertRaisesRegex(ValueError, "Exactly one of first_half_count and ",
                           workload.split, workload_all, None, None)
    self.assertRaisesRegex(ValueError, "Exactly one of first_half_count and ",
                           workload.split, workload_all, 2, .5)

    self.assertRaisesRegex(
        ValueError,
        "The first_half_count must be",
        workload.split,
        workload_all,
        first_half_count=-1)

    self.assertRaisesRegex(
        ValueError,
        "The first_half_fraction must be",
        workload.split,
        workload_all,
        first_half_fraction=-1)
    self.assertRaisesRegex(
        ValueError,
        "The first_half_fraction must be",
        workload.split,
        workload_all,
        first_half_fraction=3)

  def test_split(self):
    workload_generator = workload.WorkloadGenerator(
        test_util.QUERY_EXECUTION_DATA, test_util.TEST_SEED)
    workload_all = workload_generator.all()

    workload_train_count, workload_test_count = workload.split(
        workload_all, first_half_count=3)
    self.assertEqual(workload_train_count.query_id, test_util.TEST_QUERY_ID)
    self.assertLen(workload_train_count.query_log, 3)
    self.assertEqual(workload_all.query_log[:3], workload_train_count.query_log)

    self.assertEqual(workload_test_count.query_id, test_util.TEST_QUERY_ID)
    self.assertLen(workload_test_count.query_log, 2)
    self.assertEqual(workload_all.query_log[3:], workload_test_count.query_log)

    workload_train_fraction, workload_test_fraction = workload.split(
        workload_all, first_half_fraction=.25)
    self.assertEqual(workload_train_fraction.query_id, test_util.TEST_QUERY_ID)
    self.assertLen(workload_train_fraction.query_log, 1)
    self.assertEqual(workload_all.query_log[:1],
                     workload_train_fraction.query_log)

    self.assertEqual(workload_test_fraction.query_id, test_util.TEST_QUERY_ID)
    self.assertLen(workload_test_fraction.query_log, 4)
    self.assertEqual(workload_all.query_log[1:],
                     workload_test_fraction.query_log)


if __name__ == "__main__":
  absltest.main()

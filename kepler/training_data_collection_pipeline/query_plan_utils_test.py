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

"""Tests for query_plan_utils."""

import json
import os

from google3.pyglib import resources
from kepler.training_data_collection_pipeline import query_plan_utils
from absl.testing import absltest

_TEST_DATA_DIR = "kepler/training_data_collection_pipeline/testdata"


class QueryPlanUtilsTest(absltest.TestCase):

  def test_filter_plan_keys(self):
    remove_keys = [
        "Parallel Aware", "Relation Name", "Startup Cost",
        "Parent Relationship", "Inner Unique", "Join Filter", "Filter",
        "Plan Rows", "Plan Width", "Total Cost"
    ]

    plan_path = os.path.join(_TEST_DATA_DIR,
                             "generate_candidates_explain_plans.json")
    test_query_explain_plans = json.loads(resources.GetResource(plan_path))

    filtered_plan_path = os.path.join(
        _TEST_DATA_DIR, "filtered_generate_candidates_explain_plans.json")
    expected_filtered_query_explain_plans = json.loads(
        resources.GetResource(filtered_plan_path))

    filtered_plans = query_plan_utils.filter_keys(test_query_explain_plans,
                                                  remove_keys)
    self.assertEqual(filtered_plans, expected_filtered_query_explain_plans)


if __name__ == "__main__":
  absltest.main()

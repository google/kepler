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

"""Tests for utils."""
from typing import Any, List
from absl.testing import absltest
from absl.testing import parameterized

from kepler.training_data_collection_pipeline import query_text_utils


class QueryManagerPostgresTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="trivial no-op",
          query_template="SELECT foo FROM bar where x > 0",
          params=[],
          expected="SELECT foo FROM bar where x > 0"),
      dict(
          testcase_name="extra params no-op",
          query_template="SELECT foo FROM bar where x > 0",
          params=[1],
          expected="SELECT foo FROM bar where x > 0"),
      dict(
          testcase_name="insufficient params no-op",
          query_template="SELECT foo FROM bar where x > @param0",
          params=[],
          expected="SELECT foo FROM bar where x > @param0"),
      dict(
          testcase_name="simple substitution",
          query_template="SELECT foo FROM bar where x > @param0",
          params=[1],
          expected="SELECT foo FROM bar where x > 1"),
      # The query template with @param1 and @param10 demonstrates that the
      # direction of the naive param# replacement matters and needs to be done
      # in reverse to avoid false matches.
      dict(
          testcase_name="complex substitution",
          query_template=(
              "SELECT foo FROM bar where x > @param0 and x > @param1 and "
              "x > @param2 and x > @param3 and x > @param4 and x > @param5 and "
              "x > @param6 and x > @param7 and x > @param8 and x > @param9 and "
              "x > @param10"),
          params=list(range(1, 12)),
          expected=(
              "SELECT foo FROM bar where x > 1 and x > 2 and x > 3 and x > 4 "
              "and x > 5 and x > 6 and x > 7 and x > 8 and x > 9 and x > 10 "
              "and x > 11")))
  def test_substitute_query_params(self, query_template: str, params: List[Any],
                                   expected: str):
    """Verifies query template parameter substitution.

    Args:
      query_template: The query template that may contain 0 to many
        parameterized predicates.
      params: The parameter values to substitute into the query.
      expected: The resulting query instance with the params substituted in when
        possible.
    """
    self.assertEqual(
        query_text_utils.substitute_query_params(query_template, params),
        expected)

  def test_get_hinted_query(self):
    hints = "/*+ SeqScan(foo) */"
    query = "SELECT x from foo"
    self.assertEqual(
        query_text_utils.get_hinted_query(query=query, hints=hints),
        "/*+ SeqScan(foo) */ SELECT x from foo")


if __name__ == "__main__":
  absltest.main()

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

"""Tests for query_parsing_utils."""
from typing import Any, List


from kepler.database_integrations.model_serving import query_parsing_utils
from absl.testing import absltest
from absl.testing import parameterized


class QueryParsingUtilsTest(parameterized.TestCase):

  def test_extract_comment_content(self):
    self.assertEqual(
        query_parsing_utils.extract_comment_content("/*+ q12_1 */"), "q12_1"
    )
    self.assertEqual(
        query_parsing_utils.extract_comment_content("/*+q12_2 */"), "q12_2"
    )
    self.assertEqual(
        query_parsing_utils.extract_comment_content("/*+ q12_3*/"), "q12_3"
    )
    self.assertEqual(
        query_parsing_utils.extract_comment_content("/*+q12_4*/"), "q12_4"
    )
    self.assertEqual(
        query_parsing_utils.extract_comment_content("/*+ hi there*/"),
        "hi there",
    )
    self.assertIsNone(query_parsing_utils.extract_comment_content("hi there"))

  @parameterized.named_parameters(
      dict(
          testcase_name="0 params",
          query_template="SELECT foo FROM bar where x > 0",
          query_instance="SELECT foo FROM bar where x > 0",
          expected=[],
      ),
      dict(
          testcase_name="some params with hint to ignore",
          query_template=(
              "/*+ q11_0 */ SELECT foo FROM bar where x > @param0 and y ="
              " '@param1' and z in ('@param2')"
          ),
          query_instance=(
              "SELECT foo FROM bar where x > 5 and y = '2022-01-01' and z in"
              " ('bye')"
          ),
          expected=[5, "2022-01-01", "bye"],
      ),
      dict(
          testcase_name="out-of-order params",
          query_template=(
              "SELECT foo FROM bar where x > @param2 and y = '@param0' and z in"
              " ('@param1')"
          ),
          query_instance=(
              "SELECT foo FROM bar where x > 5 and y = '2022-01-01' and z in"
              " ('bye')"
          ),
          expected=["2022-01-01", "bye", 5],
      ),
      dict(
          testcase_name="double digit params",
          query_template=(
              "SELECT foo FROM bar where x > @param0 and x > @param10 and "
              "x > @param2 and x > @param3 and x > @param4 and x > @param5 and "
              "x > @param6 and x > @param7 and x > @param8 and x > @param9 and "
              "x > @param1"
          ),
          query_instance=(
              "SELECT foo FROM bar where x > 1 and x > 11 and x > 3 and x > 4 "
              "and x > 5 and x > 6 and x > 7 and x > 8 and x > 9 and x > 10 "
              "and x > 2"
          ),
          expected=list(range(1, 12)),
      ),
  )
  def test_extract_params(self, query_template: str, query_instance: str,
                          expected: List[Any]):
    param_extractor = query_parsing_utils.ParamExtractor(query_template)
    self.assertEqual(param_extractor.get_params(query_instance), expected)

  def test_bad_query_template(self):
    self.assertRaisesRegex(ValueError, "Param indices are not consecutive",
                           query_parsing_utils.ParamExtractor,
                           "SELECT foo from bar where x > @param1")
    self.assertRaisesRegex(
        ValueError, "Param indices are not consecutive",
        query_parsing_utils.ParamExtractor,
        "SELECT foo from bar where x > @param0 and x > @param2")
    self.assertRaisesRegex(
        ValueError, "Param indices are not consecutive starting with 0",
        query_parsing_utils.ParamExtractor,
        "SELECT foo from bar where x > @param1 and x > @param2")

  def test_bad_query_instance(self):
    param_extractor = query_parsing_utils.ParamExtractor(
        "SELECT foo from bar where x > @param0")
    self.assertRaisesRegex(ValueError, "Mismatch in flattened query tree size",
                           param_extractor.get_params,
                           "SELECT foo from bar where x > @param0 and 1=1")


if __name__ == "__main__":
  absltest.main()

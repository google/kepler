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

"""Tests for parameter_generator.py."""
import datetime
import itertools
import json
from typing import Any, List, Tuple

from kepler.training_data_collection_pipeline import parameter_generator
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util
from absl.testing import absltest

JSON = Any

# SQL keywords GROUP BY, ORDER BY, DESC, and LIMIT contain inconsistent casing
# to show that the ParameterGenerator is not case-sensitive.
_TEST_TEMPLATE_STRING = """
{
  "query": "SELECT x, \\n y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE \\n a < 2 \\n and bar.c = '@param0' \\n GrOUP BY x,y,a,b,d_date \\n ORDEr BY x \\n DeSC  \\n LiMIT 10;",
  "predicates": [
    {
      "alias": "bar",
      "column": "c",
      "operator": "="
    }
  ]
}
"""

_TEST_TEMPLATE_WEBSITE_URL = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE \\n a < 2 \\n and bar.website_url = '@param0';",
  "predicates": [
    {
      "alias": "bar",
      "column": "website_url",
      "operator": "LIKE"
    }
  ]
}
"""

_TEST_TEMPLATE_INTEGER = """
{
  "query": "SELECT j, k, a, b, d_date \\n FROM \\n baz as baz \\n JOIN \\n bar as bar \\n on j = b WHERE a > 0 \\n and baz.j = '@param0' \\n and baz.k = '@param1';",
  "predicates": [
    {
      "alias": "baz",
      "column": "j",
      "operator": "="
    },
    {
      "alias": "baz",
      "column": "k",
      "operator": "="
    }
  ]
}
"""

_TEST_TEMPLATE_LESS_THAN_OR_EQUALS = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and y <= @param0;",
  "predicates": [
    {
      "alias": "foo",
      "column": "y",
      "operator": "<="
    }
  ]
}
"""

_TEST_TEMPLATE_GREATER_THAN_OR_EQUALS = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and y >= @param0;",
  "predicates": [
    {
      "alias": "foo",
      "column": "y",
      "operator": ">="
    }
  ]
}
"""

_TEST_TEMPLATE_LESS_THAN = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and y < @param0;",
  "predicates": [
    {
      "alias": "foo",
      "column": "y",
      "operator": "<"
    }
  ]
}
"""

_TEST_TEMPLATE_GREATER_THAN = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and d_date > '@param0';",
  "predicates": [
    {
      "alias": "bar",
      "column": "d_date",
      "operator": ">"
    }
  ]
}
"""

_TEST_TEMPLATE_BETWEEN_DATE = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and bar.d_date >= '@param0' \\n and bar.d_date <= '@param1';",
  "predicates": [
    {
      "alias": "bar",
      "column": "d_date",
      "operator": ">="
    },
    {
      "alias": "bar",
      "column": "d_date",
      "operator": "<="
    }
  ]
}
"""

_TEST_TEMPLATE_BETWEEN_DATE_EXPECTED_HELPER = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and bar.d_date >= '@param0';",
  "predicates": [
    {
      "alias": "bar",
      "column": "d_date",
      "operator": ">="
    }
  ]
}
"""

_TEST_TEMPLATE_BETWEEN_TIMESTAMP = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and bar.e_date >= '@param0' \\n and bar.e_date <= '@param1';",
  "predicates": [
    {
      "alias": "bar",
      "column": "e_date",
      "operator": ">="
    },
    {
      "alias": "bar",
      "column": "e_date",
      "operator": "<="
    }
  ]
}
"""

_TEST_TEMPLATE_BETWEEN_TIMESTAMP_EXPECTED_HELPER = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and bar.e_date >= '@param0';",
  "predicates": [
    {
      "alias": "bar",
      "column": "e_date",
      "operator": ">="
    }
  ]
}
"""

_TEST_TEMPLATE_TWO_BETWEEN_INTEGERS = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and foo.y >= '@param0' \\n and foo.y <= '@param1' \\n and foo.x >= '@param2' \\n and foo.x <= '@param3';",
  "predicates": [
    {
      "alias": "foo",
      "column": "y",
      "operator": ">="
    },
    {
      "alias": "foo",
      "column": "y",
      "operator": "<="
    },
    {
      "alias": "foo",
      "column": "x",
      "operator": ">="
    },
    {
      "alias": "foo",
      "column": "x",
      "operator": "<="
    }
  ]
}
"""

_TEST_TEMPLATE_TWO_BETWEEN_INTEGERS_EXPECTED_HELPER = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and foo.y >= '@param0' \\n and foo.x >= '@param1';",
  "predicates": [
    {
      "alias": "foo",
      "column": "y",
      "operator": ">="
    },
    {
      "alias": "foo",
      "column": "x",
      "operator": ">="
    }
  ]
}
"""

_TEST_TEMPLATE_BETWEEN_DATE_WITH_ADDITIONAL_PREDICATES = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and bar.a = @param0 \\n and bar.d_date >= '@param1' \\n and bar.d_date <= '@param2';",
  "predicates": [
    {
      "alias": "bar",
      "column": "a",
      "operator": "="
    },
    {
      "alias": "bar",
      "column": "d_date",
      "operator": ">="
    },
    {
      "alias": "bar",
      "column": "d_date",
      "operator": "<="
    }
  ]
}
"""

_TEST_TEMPLATE_BETWEEN_DATE_WITH_ADDITIONAL_PREDICATES_EXPECTED_HELPER = """
{
  "query": "SELECT x, y, a, b, d_date \\n FROM \\n foo as foo \\n JOIN \\n bar as bar \\n on x = b WHERE a > 0 \\n and bar.a = @param0 \\n and bar.d_date >= '@param1';",
  "predicates": [
    {
      "alias": "bar",
      "column": "a",
      "operator": "="
    },
    {
      "alias": "bar",
      "column": "d_date",
      "operator": ">="
    }
  ]
}
"""


def _get_template_item(
    query_id: str,
    template_serialized: str) -> parameter_generator.TemplateItem:
  return parameter_generator.TemplateItem(
      query_id=query_id, template=json.loads(template_serialized))


class ParameterGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    database_configuration = query_utils.DatabaseConfiguration(
        dbname=self._test_database.dbname,
        user=test_util.USER,
        password=test_util.PASSWORD)
    self._query_manager = query_utils.QueryManager(database_configuration)
    test_util.populate_database(self._query_manager)
    self._parameter_generator = parameter_generator.ParameterGenerator(
        database_configuration)

  def tearDown(self):
    self._test_database.drop()
    super().tearDown()

  def _get_possible_parameter_values(self, template: JSON) -> List[Tuple[JSON]]:
    """Identifies all parameter values that produce results for the query.

    Args:
      template: Query template for which to get possible parameter values.

    Returns:
      List of lists of parameter values, with each sublist representing the
      one complete set of parameter values that produces a result.
    """
    parameter_candidates = parameter_generator._get_distinct_column_values(
        template['predicates'], template, self._query_manager)

    parameter_values = []
    for params in itertools.product(*parameter_candidates):
      # Apply predicate operator adjustments to values.
      adjusted_params = []
      for param, predicate in zip(params, template['predicates']):
        if isinstance(param, datetime.date):
          adjusted_params.append(
              parameter_generator._format_date(
                  param + parameter_generator._get_date_operator_adjustment(
                      predicate['operator'])))
          continue

        if isinstance(param, str):
          adjusted_params.append(param)
          continue

        adjusted_params.append(
            param + parameter_generator._get_numeric_operator_adjustment(
                predicate['operator']))

      adjusted_params = tuple(adjusted_params)
      if self._query_manager.execute(
          query=template['query'], params=adjusted_params):
        parameter_values.append(adjusted_params)

    return parameter_values

  def test_single_predicate_per_column(self):
    template_test_cases = [
        _TEST_TEMPLATE_STRING, _TEST_TEMPLATE_INTEGER,
        _TEST_TEMPLATE_LESS_THAN_OR_EQUALS,
        _TEST_TEMPLATE_GREATER_THAN_OR_EQUALS, _TEST_TEMPLATE_LESS_THAN,
        _TEST_TEMPLATE_GREATER_THAN
    ]

    for template in template_test_cases:
      template_item = _get_template_item(test_util.TEST_QUERY_ID, template)
      expected = set(
          self._get_possible_parameter_values(template_item.template))
      for count in [1, len(expected)]:
        output = self._parameter_generator.generate_parameters(
            count=count, template_item=template_item)
        actual = set(
            [tuple(x) for x in output[test_util.TEST_QUERY_ID]['params']])
        self.assertLen(actual, count)
        self.assertTrue(actual.issubset(expected))

  def test_single_predicate_website_url(self):
    template_item = _get_template_item(test_util.TEST_QUERY_ID,
                                       _TEST_TEMPLATE_WEBSITE_URL)
    expected = set([
        parameter_generator._extract_url_domain(url[0])
        for url in self._get_possible_parameter_values(template_item.template)
    ])

    for count in [1, len(expected)]:
      output = self._parameter_generator.generate_parameters(
          count=count, template_item=template_item)
      actual = set(
          [tuple(x)[0] for x in output[test_util.TEST_QUERY_ID]['params']])
      self.assertLen(actual, count)
      self.assertTrue(actual.issubset(expected))

  def test_range_predicate_on_column_date(self):
    actual_template_item = _get_template_item(test_util.TEST_QUERY_ID,
                                              _TEST_TEMPLATE_BETWEEN_DATE)
    expected_template_item = _get_template_item(
        test_util.TEST_QUERY_ID, _TEST_TEMPLATE_BETWEEN_DATE_EXPECTED_HELPER)
    expected_helper = self._get_possible_parameter_values(
        expected_template_item.template)

    self.assertLen(expected_template_item.template['predicates'], 1)

    output = self._parameter_generator.generate_parameters(
        count=len(expected_helper), template_item=actual_template_item)
    actual = [tuple(x) for x in output[test_util.TEST_QUERY_ID]['params']]

    # The first two values represent the lower and upper bound for d_date. The
    # expected values at the center of each range are respectively
    # [['2021-09-19'], ['2021-09-20'], ['2021-09-21']]. The bounds are chosen
    # from [column_min - 1 day, value) and (value, column_max + 1 day] for lower
    # and upper respectively. For d_date, column_min = '2021-09-19' and
    # column_max = '2021-09-22'.
    expected = [('2021-09-18', '2021-09-23'), ('2021-09-18', '2021-09-22'),
                ('2021-09-20', '2021-09-23')]
    self.assertEqual(actual, expected)

  def test_range_predicate_on_column_timestamp(self):
    """Verifies that timestamps are cast to dates when generating params."""
    actual_template_item = _get_template_item(test_util.TEST_QUERY_ID,
                                              _TEST_TEMPLATE_BETWEEN_TIMESTAMP)
    expected_template_item = _get_template_item(
        test_util.TEST_QUERY_ID,
        _TEST_TEMPLATE_BETWEEN_TIMESTAMP_EXPECTED_HELPER)
    expected_helper = self._get_possible_parameter_values(
        expected_template_item.template)

    self.assertLen(expected_template_item.template['predicates'], 1)

    # Request one more parameter than we expect to ensure an extra one isn't
    # produced.
    output = self._parameter_generator.generate_parameters(
        count=len(expected_helper) + 1, template_item=actual_template_item)
    actual = [tuple(x) for x in output[test_util.TEST_QUERY_ID]['params']]

    # There are multiple timestamps on 2/7/2022 which should get merged together
    # to leave only two distinct values. The actual vs expected check at the end
    # verifies that len(actual) == 2.
    self.assertLen(actual, len(expected_helper))

    # The bounds are chosen from [column_min - 1 day, value) and (value,
    # column_max + 1 day] for lower and upper respectively. For e_date,
    # column_min = '2022-02-07' and column_max = '2022-02-08'.
    expected = [('2022-02-07', '2022-02-09'), ('2022-02-06', '2022-02-09')]
    self.assertEqual(actual, expected)

  def test_multiple_range_predicate_on_column_integer(self):
    actual_template_item = _get_template_item(
        test_util.TEST_QUERY_ID, _TEST_TEMPLATE_TWO_BETWEEN_INTEGERS)
    expected_template_item = _get_template_item(
        test_util.TEST_QUERY_ID,
        _TEST_TEMPLATE_TWO_BETWEEN_INTEGERS_EXPECTED_HELPER)
    expected_helper = self._get_possible_parameter_values(
        expected_template_item.template)

    self.assertLen(expected_template_item.template['predicates'], 2)

    output = self._parameter_generator.generate_parameters(
        count=len(expected_helper), template_item=actual_template_item)
    actual = [tuple(x) for x in output[test_util.TEST_QUERY_ID]['params']]

    # The first two values represent the lower and upper bound for y. The
    # expected values at the center of each range are respectively 4, 2, 3, 2,
    # -1. The bounds are chosen from [column_min -1, value) and (value,
    # column_max + 1] for lower and upper respectively. For y, column_min = -1
    # and column_max = 4.
    #
    # The latter two values represent the lower and upper bound for x. The
    # expected values at the center of each range are respectively 2, 2, 1, 1,
    # 1. The bounds are chosen from [column_min - 1, value) and (value,
    # column_max + 1] for lower and upper respectively. For x, column_min = 1
    # and column_max = 2.
    expected = [(3, 5, 0, 3), (-1, 4, 1, 3), (-1, 4, 0, 3), (-2, 3, 0, 2),
                (-2, 2, 0, 3)]
    self.assertEqual(actual, expected)

  def test_range_predicate_on_column_date_distinct_ranges(self):
    """Verify logic that ensures ranges are distinct.

    If the rest of the parameter values are the same, then ranges must be
    distinct as well. Due to implementation details, distinct ranges are
    enforced by a different code path than keeping the other parameters
    distinct.

    This test sweeps across a range of seeds to ensure no seed causes an overlap
    in ranges to be produced. This ensures the test passing is not dependent on
    luck and a specific seed value.
    """
    for i in range(1000):
      local_parameter_generator = parameter_generator.ParameterGenerator(
          query_utils.DatabaseConfiguration(
              dbname=self._test_database.dbname,
              user=test_util.USER,
              password=test_util.PASSWORD,
              seed=i / 1000.))

      actual_template_item = _get_template_item(
          test_util.TEST_QUERY_ID,
          _TEST_TEMPLATE_BETWEEN_DATE_WITH_ADDITIONAL_PREDICATES)
      expected_template_item = _get_template_item(
          test_util.TEST_QUERY_ID,
          _TEST_TEMPLATE_BETWEEN_DATE_WITH_ADDITIONAL_PREDICATES_EXPECTED_HELPER
      )
      expected_helper = self._get_possible_parameter_values(
          expected_template_item.template)

      self.assertLen(expected_template_item.template['predicates'], 2)

      output = local_parameter_generator.generate_parameters(
          count=len(expected_helper), template_item=actual_template_item)
      actual = [tuple(x) for x in output[test_util.TEST_QUERY_ID]['params']]

      self.assertLen(actual, len(set(actual)))

  def test_dry_run(self):
    actual_template_item = _get_template_item(test_util.TEST_QUERY_ID,
                                              _TEST_TEMPLATE_INTEGER)
    expected_template_item = _get_template_item(test_util.TEST_QUERY_ID,
                                                _TEST_TEMPLATE_INTEGER)
    expected = self._get_possible_parameter_values(
        expected_template_item.template)
    self.assertLen(expected, 4)

    output = self._parameter_generator.generate_parameters(
        count=len(expected), template_item=actual_template_item, dry_run=True)
    actual = [tuple(x) for x in output[test_util.TEST_QUERY_ID]['params']]

    # A single parameter is generated in dry-run mode. There is no guarantee
    # regarding which of the possible parameters will be generated.
    self.assertLen(actual, 1)
    self.assertIn(actual[0], expected)

if __name__ == '__main__':
  absltest.main()

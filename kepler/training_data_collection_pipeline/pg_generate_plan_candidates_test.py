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

import functools
from typing import Any, List, Tuple, Union
from unittest import mock

import numpy as np

from kepler.training_data_collection_pipeline import pg_generate_plan_candidates
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util
from absl.testing import absltest
from absl.testing import parameterized


_TEST_QUERY_0 = """
SELECT x, y, a, b, d_date \n FROM \n foo JOIN bar on x = b WHERE \n a < 2 \n and bar.c = '@param0';
"""

_TEST_QUERY_1 = """
SELECT foo.x, bar.a \n FROM foo, bar, baz\n WHERE \n bar.a < 1 \n AND bar.c = '@param0'\n AND bar.b < 3\n AND baz.j = 1\n AND foo.x = baz.j AND baz.k < 4\n AND baz.k > 2;
"""

_TEST_QUERY_2 = """
SELECT foo.x, bar.a \n FROM foo, bar, baz\n WHERE \n bar.a < 1 \n AND bar.c = '@param0'\n AND bar.b < @param1\n AND baz.j = 1\n AND foo.x = baz.j AND baz.k < 4\n AND baz.k > 2;
"""

JSON = Any


def _generate_dummy_explain(identifier_a: Union[str, int],
                            identifier_b: Union[str, int]) -> JSON:
  """Generates a basic EXPLAIN JSON identified by two parameters.

  Args:
    identifier_a: First value to identify plan.
    identifier_b: Second value to identify plan.

  Returns:
    Dummy JSON containing scans over fake tables a_b_1 and a_b_2.
  """
  return {
      'Plan': {
          'Join Type':
              'Inner',
          'Node Type':
              'Nested Loop',
          'Plans': [{
              'Parent Relationship': 'Outer',
              'Node Type': 'Seq Scan',
              'Alias': f'{identifier_a}_{identifier_b}_1'
          }, {
              'Parent Relationship': 'Inner',
              'Node Type': 'Seq Scan',
              'Alias': f'{identifier_a}_{identifier_b}_2'
          }]
      }
  }


def _dummy_generation(params: List[str],
                      default_only: bool = False,
                      **kwargs) -> JSON:
  """Generates dummy candidate plans based on a single int parameter.

  For parameter value x, generates a default plan and x additional plans,
  each uniquely identifiable via _generate_dummy_explain.

  Args:
    params: Parameters for this query.
    default_only: Whether to only generate the default plan or not.
    **kwargs: Unused, only here to mock with pg_generate_plan_candidates
      execute_plan_generation.

  Returns:
    Standard pg_generate_plan_candidates output dict.
  """
  del kwargs  # See above.
  num = int(params[0])
  additional_plans = [] if default_only else [
      _generate_dummy_explain(num, i) for i in range(num)
  ]
  sources = [] if default_only else [f'{num}_{i}' for i in range(num)]
  return {
      'params': params,
      'result': _generate_dummy_explain('default', num),
      'additional_plans': additional_plans,
      'sources': sources,
  }


class GeneratePlanCandidatesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._database_configuration = query_utils.DatabaseConfiguration(
        dbname=self._test_database.dbname,
        user=test_util.USER,
        password=test_util.PASSWORD)
    self._query_manager = query_utils.QueryManager(self._database_configuration)
    test_util.populate_database(self._query_manager)

  def tearDown(self):
    self._test_database.drop()
    super().tearDown()

  def test_get_query_plans_with_configs(self):
    params = ['https://hello.com']
    configs = ['enable_nestloop', 'enable_hashjoin']
    result = pg_generate_plan_candidates.get_query_plans(
        params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_0,
        configs=configs)
    self.check_result_with_configs(result, params)

  def test_get_query_plans_valid_join_configs(self):
    params = ['https://hello.com']
    configs = ['enable_nestloop', 'enable_hashjoin', 'enable_mergejoin']
    result = pg_generate_plan_candidates.get_query_plans(
        params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_0,
        configs=configs)
    self.assertLen(result['additional_plans'], 6)
    self.assertLen(result['sources'], 6)
    for source in result['sources']:
      self.assertLess(len(source), 3)

  def test_get_query_plans_valid_scan_configs(self):
    params = ['https://hello.com']
    configs = ['enable_indexonlyscan', 'enable_indexscan', 'enable_seqscan']
    result = pg_generate_plan_candidates.get_query_plans(
        params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_0,
        configs=configs)
    self.assertLen(result['additional_plans'], 6)
    self.assertLen(result['sources'], 6)
    for source in result['sources']:
      self.assertLess(len(source), 3)

  def test_get_query_plans_all_valid_configs(self):
    params = ['https://hello.com']
    configs = ['enable_nestloop', 'enable_hashjoin', 'enable_mergejoin',
               'enable_indexonlyscan', 'enable_indexscan', 'enable_seqscan']
    result = pg_generate_plan_candidates.get_query_plans(
        params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_0,
        configs=configs)
    self.assertLen(result['additional_plans'], 48)
    self.assertLen(result['sources'], 48)

  def test_get_query_plans_with_no_configs(self):
    params = ['https://hello.com']
    configs = []
    result = pg_generate_plan_candidates.get_query_plans(
        params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_0,
        configs=configs)
    additional_plans = result['additional_plans']
    sources = result['sources']

    self.assertEqual(result['params'], params)
    self.assertIn('Plan', result['result'])
    self.assertEmpty(additional_plans)
    self.assertEmpty(sources)

  def test_execute_plan_generation_distributed(self):
    # Repeat some params to decrease probability that plan generation
    # shuffling the params passes this test.
    params = [['https://hello.com'],
              ['http://goodbye.org/methods'],
              ['http://www.goodnight.org'],
              ['http://www.goodmorning.com'],
              ['https://hello.com'],
              ['http://www.goodmorning.com'],
              ['http://goodbye.org/methods'],
              ['https://hello.com'],
              ['http://www.goodnight.org']] * 2
    configs = ['enable_nestloop', 'enable_hashjoin']
    query_metadata = {'query': _TEST_QUERY_0, 'params': params}

    function_kwargs = {
        'database_configuration': self._database_configuration,
        'query': query_metadata['query'],
        'configs': configs
    }
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_generate_plan_candidates.execute_plan_generation(
        pg_generate_plan_candidates.get_query_plans,
        function_kwargs,
        query_metadata['params'],
        plan_hint_extractor=plan_hint_extractor,
        chunksize=1,
        distributed=True)

    hint_counts, _, _, _ = plan_hint_extractor.get_consolidated_plan_hints()
    self.check_extracted_hints(
        actual_hint_counts=hint_counts,
        expected_identifying_token_in_hints=[
            'MergeJoin', 'NestLoop', 'HashJoin'
        ],
        expected_counts=[18, 18, 36],
        expected_sources=[
            tuple(['enable_nestloop', 'enable_hashjoin']),
            tuple(['enable_hashjoin']),
            'default',
        ])

  @parameterized.named_parameters(
      dict(
          testcase_name='limit7_distributed',
          limit=7,
          distributed=True,
          expected_num_plans=8,
          expected_tokens=[
              '2_0_1', '2_1_1', '3_0_1', '3_1_1', '3_2_1', 'default_1',
              'default_2', 'default_3'
          ],
          expected_sources=[
              '2_0', '2_1', '3_0', '3_1', '3_2', 'default', 'default', 'default'
          ]),
      dict(
          testcase_name='limit8_distributed',
          limit=8,
          distributed=True,
          expected_num_plans=9,
          expected_tokens=[
              '1_0_1', '2_0_1', '2_1_1', '3_0_1', '3_1_1', '3_2_1', 'default_1',
              'default_2', 'default_3'
          ],
          expected_sources=[
              '1_0', '2_0', '2_1', '3_0', '3_1', '3_2', 'default', 'default',
              'default'
          ]),
      dict(
          testcase_name='no_limit_distributed',
          limit=None,
          distributed=True,
          expected_num_plans=9,
          expected_tokens=[
              '1_0_1', '2_0_1', '2_1_1', '3_0_1', '3_1_1', '3_2_1', 'default_1',
              'default_2', 'default_3'
          ],
          expected_sources=[
              '1_0', '2_0', '2_1', '3_0', '3_1', '3_2', 'default', 'default',
              'default'
          ]),
      dict(
          testcase_name='limit7_serial',
          limit=7,
          distributed=False,
          expected_num_plans=8,
          expected_tokens=[
              '2_0_1', '2_1_1', '3_0_1', '3_1_1', '3_2_1', 'default_1',
              'default_2', 'default_3'
          ],
          expected_sources=[
              '2_0', '2_1', '3_0', '3_1', '3_2', 'default', 'default', 'default'
          ]),
      dict(
          testcase_name='limit8_serial',
          limit=8,
          distributed=False,
          expected_num_plans=9,
          expected_tokens=[
              '1_0_1', '2_0_1', '2_1_1', '3_0_1', '3_1_1', '3_2_1', 'default_1',
              'default_2', 'default_3'
          ],
          expected_sources=[
              '1_0', '2_0', '2_1', '3_0', '3_1', '3_2', 'default', 'default',
              'default'
          ]),
      dict(
          testcase_name='no_limit_serial',
          limit=None,
          distributed=False,
          expected_num_plans=9,
          expected_tokens=[
              '1_0_1', '2_0_1', '2_1_1', '3_0_1', '3_1_1', '3_2_1', 'default_1',
              'default_2', 'default_3'
          ],
          expected_sources=[
              '1_0', '2_0', '2_1', '3_0', '3_1', '3_2', 'default', 'default',
              'default'
          ]),
  )
  @mock.patch(('kepler.training_data_collection_pipeline.'
               'pg_generate_plan_candidates.get_query_plans'),
              functools.partial(_dummy_generation, default_only=True))
  def test_execute_plan_generation_limit_total(self, limit, distributed,
                                               expected_num_plans,
                                               expected_tokens,
                                               expected_sources):
    """Tests limiting total number of plans via soft_total_plans_limit."""
    # Should produce 4, 3, 2 plans respectively (with no constraints).
    params = [[str(i)] for i in range(3, 0, -1)]
    query_metadata = {'query': _TEST_QUERY_0, 'params': params}

    function_kwargs = {}
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_generate_plan_candidates.execute_plan_generation(
        _dummy_generation,
        function_kwargs,
        query_metadata['params'],
        plan_hint_extractor=plan_hint_extractor,
        chunksize=1,
        distributed=distributed,
        soft_total_plans_limit=limit)

    hint_counts, _, _, _ = plan_hint_extractor.get_consolidated_plan_hints()
    self.check_extracted_hints(
        actual_hint_counts=hint_counts,
        expected_identifying_token_in_hints=expected_tokens,
        expected_counts=[1 for i in range(expected_num_plans)],
        expected_sources=expected_sources)

  def test_row_number_evolution(self):
    params = ['https://hello.com']
    np.random.seed(0)
    result = pg_generate_plan_candidates.generate_by_row_num_evolution(
        params=params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_1,
        num_generations=3,
        num_mutations_per_plan=2)
    self.assertEqual(result['params'], params)
    self.assertIn('Plan', result['result'])
    self.assertLen(result['additional_plans'], 1)
    self.assertNotEqual(result['result'], result['additional_plans'][0])
    self.assertLen(result['sources'], 1)
    self.assertIn('Rows', result['sources'][0])

  @parameterized.named_parameters(
      dict(testcase_name='limit1', max_plans=1, expected_plans=1),
      dict(testcase_name='limit2', max_plans=2, expected_plans=2),
  )
  def test_row_number_evolution_max_plans(self, max_plans, expected_plans):
    """Test row num evolution max_plans parameter."""
    params = ['https://hello.com']
    np.random.seed(0)
    result = pg_generate_plan_candidates.generate_by_row_num_evolution(
        params=params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_1,
        max_plans=max_plans,
        num_generations=3,
        num_mutations_per_plan=2,
        perturb_unit_only=False)
    self.assertEqual(result['params'], params)
    self.assertIn('Plan', result['result'])
    self.assertLen(result['additional_plans'], expected_plans)
    self.assertNotEqual(result['result'], result['additional_plans'][0])
    self.assertLen(result['sources'], expected_plans)
    self.assertIn('Rows', result['sources'][0])

  def test_row_number_evolution_perturb_all(self):
    """Test row num evolution perturbing all estimates."""
    params = ['https://hello.com']
    np.random.seed(0)
    result = pg_generate_plan_candidates.generate_by_row_num_evolution(
        params=params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_1,
        num_generations=3,
        num_mutations_per_plan=2,
        perturb_unit_only=False)
    self.assertEqual(result['params'], params)
    self.assertIn('Plan', result['result'])
    # Perturbing not just unit row count estimates leads to more plans.
    self.assertLen(result['additional_plans'], 2)
    self.assertLen(result['sources'], 2)
    for i in range(2):
      self.assertNotEqual(result['result'], result['additional_plans'][i])
      self.assertIn('Rows', result['sources'][i])

    self.assertEqual(result['debug_info']['candidates_per_generation'],
                     [1, 1, 1, 0])

  @parameterized.named_parameters(
      dict(testcase_name='max1', max_perturbs_per_join=1, expected_num_plans=3),
      dict(testcase_name='max2', max_perturbs_per_join=2, expected_num_plans=4),
      dict(testcase_name='no_max', max_perturbs_per_join=None,
           expected_num_plans=4)
  )
  def test_row_number_evolution_perturb_count_limit(self,
                                                    max_perturbs_per_join,
                                                    expected_num_plans):
    """Tests limiting how many times each join can be perturbed."""
    params = ['https://hello.com', 1]
    np.random.seed(0)
    result = pg_generate_plan_candidates.generate_by_row_num_evolution(
        params=params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_2,
        num_generations=5,
        num_mutations_per_plan=3,
        exponent_base=7,
        exponent_range=2,
        max_plans_per_generation=3,
        perturb_unit_only=False,
        max_perturbs_per_join=max_perturbs_per_join)
    self.assertEqual(result['params'], params)
    self.assertIn('Plan', result['result'])
    self.assertLen(result['additional_plans'], expected_num_plans)
    self.assertLen(result['sources'], expected_num_plans)
    for i in range(expected_num_plans):
      self.assertNotEqual(result['result'], result['additional_plans'][i])
      self.assertIn('Rows', result['sources'][i])

  @parameterized.named_parameters(
      ('actual_1', 1, 10, 2, np.array([1, 10, 100, 1000, 10000])),
      ('truncate_left', 10, 10, 2, np.array([1, 10, 100, 1000, 10000])),
      ('truncate_left_rounding', 6, 2, 3, np.array([1, 3, 6, 12, 24, 48, 96])),
      ('no_truncation_rounding', 6, 2, 2, np.array([1, 3, 6, 12, 24])),
      (
          'no_truncation',
          1000,
          100,
          2,
          np.array([10, 1000, 100000, 10000000, 1000000000]),
      ),
  )
  def test_row_count_candidates(self, row_count, exponent_base,
                                exponent_range, expected_candidates):
    self.assertTrue(np.array_equal(
        pg_generate_plan_candidates.get_row_count_candidates(
            row_count, exponent_base, exponent_range), expected_candidates))

  @parameterized.named_parameters(
      dict(testcase_name='single', cardinality_multipliers=[10]),
      dict(testcase_name='multiple', cardinality_multipliers=[.1, 1, 10]))
  def test_generate_by_exhaustive_cardinality_perturbations(
      self, cardinality_multipliers):
    """Verifies number of plans produced by exhaustive cardinality perturbation.

    Args:
      cardinality_multipliers: Multipliers to adjust the cardinality estimate
        for each join.
    """
    params = ['https://hello.com', 1]
    np.random.seed(0)
    result = pg_generate_plan_candidates.generate_by_exhaustive_cardinality_perturbations(
        params=params,
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_2,
        cardinality_multipliers=cardinality_multipliers)
    self.assertEqual(result['params'], params)
    self.assertIn('Plan', result['result'])
    # The number of plans generated: (|cardinality_multipliers|) ^ 2 joins.
    expected_plan_count = len(cardinality_multipliers) ** 2
    self.assertLen(result['additional_plans'], expected_plan_count)
    self.assertEqual(result['sources'],
                     ['https://hello.com####1'] * expected_plan_count)

  def test_execute_plan_generation_not_distributed(self):
    params = [['https://hello.com', 1], ['https://goodbye.com', 2]]
    np.random.seed(0)

    query_metadata = {'query': _TEST_QUERY_2, 'params': params}
    function_kwargs = {
        'database_configuration': self._database_configuration,
        'query': query_metadata['query'],
        'cardinality_multipliers': [.1, 1, 10]
    }

    # The generate_by_exhaustive_cardinality_perturbations implementation has a
    # library call that spawns multiple processes. Therefore,
    # execute_plan_generation may not also use multiple processes as it does
    # when distributed=True.
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    with self.assertRaisesRegex(
        AssertionError, 'daemonic processes are not allowed to have children'):
      pg_generate_plan_candidates.execute_plan_generation(
          pg_generate_plan_candidates
          .generate_by_exhaustive_cardinality_perturbations,
          function_kwargs,
          query_metadata['params'],
          plan_hint_extractor=plan_hint_extractor,
          chunksize=1,
          distributed=True)

    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_generate_plan_candidates.execute_plan_generation(
        pg_generate_plan_candidates
        .generate_by_exhaustive_cardinality_perturbations,
        function_kwargs,
        query_metadata['params'],
        plan_hint_extractor=plan_hint_extractor,
        chunksize=1,
        distributed=False)

    hint_counts, _, _, _ = plan_hint_extractor.get_consolidated_plan_hints()

    # For each parameter, we expect (3 multipliers) ^ 2 joins = 9 plans will be
    # generated. Additionally, there will be 1 default plan resulting in 10
    # plans per parameter and 20 plans in total. Therefore, the expected_counts
    # sum to 20.
    self.check_extracted_hints(
        actual_hint_counts=hint_counts,
        expected_identifying_token_in_hints=[
            'Leading(((bar baz) foo))', 'Leading((bar (baz foo)))'
        ],
        expected_counts=[14, 6],
        expected_sources=['default', 'https://hello.com####1'])

  def check_extracted_hints(
      self, actual_hint_counts: JSON,
      expected_identifying_token_in_hints: List[str],
      expected_counts: List[int], expected_sources: List[Union[Tuple[str],
                                                               str]]) -> None:
    actual_hints = []
    actual_counts = []
    actual_sources = []
    for hint in sorted(actual_hint_counts):
      actual_hints.append(hint)
      actual_counts.append(actual_hint_counts[hint]['count'])
      actual_sources.append(actual_hint_counts[hint]['source'])

    for hint, expected_token in zip(actual_hints,
                                    expected_identifying_token_in_hints):
      self.assertIn(expected_token, hint)
    self.assertEqual(actual_counts, expected_counts)
    self.assertEqual(actual_sources, expected_sources)

  def check_result_with_configs(self, result: JSON,
                                target_params: List[str]) -> None:
    additional_plans = result['additional_plans']
    sources = result['sources']
    self.assertEqual(result['params'], target_params)
    self.assertIn('Plan', result['result'])
    for plan in additional_plans:
      self.assertIn('Plan', plan)
    self.assertNotEqual(additional_plans[0], additional_plans[1])
    self.assertNotEqual(additional_plans[0], additional_plans[2])
    self.assertNotEqual(additional_plans[1], additional_plans[2])
    self.assertLen(additional_plans, 3)
    self.assertEqual(sources, [('enable_nestloop',), ('enable_hashjoin',),
                               ('enable_nestloop', 'enable_hashjoin')])


if __name__ == '__main__':
  absltest.main()

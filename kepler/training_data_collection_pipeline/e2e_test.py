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

"""End-to-end tests for the training data collection pipeline.

The primary goal of this test is to ensure the outputs and inputs of each
pipeline step flow seamlessly into the next step for various pipeline
compositions.

The correctness of the output of each library function is verified by the unit
tests for that library function.
"""

import copy
import json
from typing import Any, List, Optional, Tuple

from kepler.training_data_collection_pipeline import parameter_generator
from kepler.training_data_collection_pipeline import pg_execute_training_data_queries
from kepler.training_data_collection_pipeline import pg_generate_plan_candidates
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util
from absl.testing import absltest
from absl.testing import parameterized

# Typing aliases.
JSON = Any

_TEST_TEMPLATE_INTEGER = """
{
  "query": "SELECT j, k, a, b, d_date \\n FROM \\n baz as baz \\n JOIN \\n bar as bar \\n on j = b \\n JOIN \\n foo as foo \\n on x = j \\n WHERE foo.y = 2 \\n and baz.j = '@param0' \\n and baz.k = '@param1';",
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

_GET_QUERY_PLANS_KWARGS = {
    "configs": ["enable_nestloop", "enable_hashjoin"],
}

_GENERATE_BY_ROW_NUM_EVOLUTION_KWARGS = {
    "num_generations": 3,
    "num_mutations_per_plan": 3,
    "exponent_base": 10,
    "exponent_range": 3,
    "max_plans_per_generation": 3,
    "perturb_unit_only": False
}

_GENERATE_BY_EXHAUSTIVE_CARDINALITY_PERTURBATION_KWARGS = {
    "cardinality_multipliers": [.1, 10],
}


class E2ETest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._database_configuration = query_utils.DatabaseConfiguration(
        dbname=self._test_database.dbname,
        user=test_util.USER,
        password=test_util.PASSWORD)
    self._query_manager = query_utils.QueryManager(self._database_configuration)
    test_util.populate_database(self._query_manager)
    self._parameter_generator = parameter_generator.ParameterGenerator(
        self._database_configuration)

  def tearDown(self):
    self._test_database.drop()
    super().tearDown()

  @parameterized.named_parameters(
      ("get_query_plans", pg_generate_plan_candidates.get_query_plans,
       _GET_QUERY_PLANS_KWARGS, 2),
      ("generate_by_row_num_evolution",
       pg_generate_plan_candidates.generate_by_row_num_evolution,
       _GENERATE_BY_ROW_NUM_EVOLUTION_KWARGS, 3),
      ("generate_by_exhaustive_cardinality_perturbations",
       pg_generate_plan_candidates
       .generate_by_exhaustive_cardinality_perturbations,
       _GENERATE_BY_EXHAUSTIVE_CARDINALITY_PERTURBATION_KWARGS, 2))
  def test_run_pipeline(self, plan_generation_function,
                        plan_generation_function_specific_kwargs,
                        extracted_plan_hints_count):
    # Step 0: Generate parameters.
    template_item = parameter_generator.TemplateItem(
        query_id=test_util.TEST_QUERY_ID,
        template=json.loads(_TEST_TEMPLATE_INTEGER))
    parameter_count = 3
    query_templates = self._parameter_generator.generate_parameters(
        count=parameter_count, template_item=template_item)
    query_metadata = query_templates[test_util.TEST_QUERY_ID]
    self.assertLen(query_metadata["params"], parameter_count)

    # Step 1: Generate plan candidates.
    plan_generation_function_kwargs = {
        "query": query_metadata["query"],
        "database_configuration": self._database_configuration
    }
    plan_generation_function_kwargs.update(
        plan_generation_function_specific_kwargs)
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_generate_plan_candidates.execute_plan_generation(
        function=plan_generation_function,
        function_kwargs=plan_generation_function_kwargs,
        all_params=query_metadata["params"],
        plan_hint_extractor=plan_hint_extractor,
        chunksize=1,
        distributed=False)
    hint_counts, plan_hints, parameter_values_and_plan_indices, debug_infos = plan_hint_extractor.get_consolidated_plan_hints(
    )
    self.assertLen(hint_counts, extracted_plan_hints_count)
    self.assertLen(plan_hints, extracted_plan_hints_count)
    self.assertLen(parameter_values_and_plan_indices, parameter_count)
    if plan_generation_function == pg_generate_plan_candidates.generate_by_row_num_evolution:
      self.assertLen(debug_infos, parameter_count)

    # Step 2: Verify plan hints.
    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=test_util.TEST_QUERY_ID,
        query=query_metadata["query"],
        plan_hints=plan_hints,
        params_plan_indices=parameter_values_and_plan_indices,
        database_configuration=self._database_configuration)
    self.assertLen(failure_counts, 1)
    self.assertLen(failure_counts[test_util.TEST_QUERY_ID],
                   extracted_plan_hints_count)

    # Step 3: Collect execution results.
    query_manager = query_utils.QueryManager(self._database_configuration)

    def execute_query(
        unused_query_manager: query_utils.QueryManager,
        query: str,
        params: List[Any],
        timeout_ms: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[int]]:
      del unused_query_manager
      return query_manager.execute_timed(query, params, timeout_ms)

    execute_query_results_key = "duration_ms"

    execution_results = {}
    execution_metadata = {}

    def checkpoint_results(query_id: str, results: Any,
                           is_metadata: bool) -> None:
      self.assertEqual(test_util.TEST_QUERY_ID, query_id)

      if is_metadata:
        nonlocal execution_metadata
        execution_metadata = copy.deepcopy(results)
      else:
        nonlocal execution_results
        execution_results = copy.deepcopy(results)

    pg_execute_training_data_queries.execute_training_data_queries(
        batch_index=0,
        parameter_values={
            test_util.TEST_QUERY_ID: parameter_values_and_plan_indices
        },
        query_id=test_util.TEST_QUERY_ID,
        templates=query_templates,
        plan_hints={test_util.TEST_QUERY_ID: plan_hints},
        iterations=3,
        batch_size=2,
        skip_indices=[],
        query_timeout_multiplier=3,
        query_timeout_min_ms=30,
        query_timeout_max_ms=300,
        execute_query_fn=execute_query,
        checkpoint_results_fn=checkpoint_results,
        results_key=execute_query_results_key,
        limit=None,
        plan_cover_num_params=2,
        near_optimal_threshold=1.05,
        num_params_threshold=0.95,
        query_timeout_minimum_speedup_multiplier=2,
    )

    self.assertLen(execution_results[test_util.TEST_QUERY_ID], parameter_count)
    self.assertLen(execution_metadata[test_util.TEST_QUERY_ID], 1)


if __name__ == "__main__":
  absltest.main()

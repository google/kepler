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

import copy
import json
import os

from typing import Any

from google3.pyglib import resources
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util
from absl.testing import absltest

_TEST_QUERY_0 = """
SELECT x, y, a, b, d_date \n FROM \n foo JOIN bar on x = b WHERE \n a < 2 \n and bar.c = '@param0';
"""
_TEST_QUERY_1 = """
SELECT x, y\n FROM \n foo WHERE \n x < 2;
"""

_TEST_QUERY_2 = """
SELECT x, y, a, b, d_date, j, k \n FROM \n foo, bar, baz\n WHERE \n a < 2 \n and bar.c = '@param0';
"""

_TEST_QUERY_3_SUBQUERY_PREDICATE_INITPLAN = """
SELECT x, y, a, b, d_date \n FROM \n foo JOIN bar on x = b WHERE \n a < 2 \n and bar.c = '@param0' \n and bar.b = (SELECT j from baz ORDER BY k LIMIT 1);
"""

_TEST_QUERY_6_SUBQUERY_PREDICATE_INITPLAN_IS_SCAN_NODE = """
SELECT a, b, d_date \n FROM \n bar WHERE \n a < 2 \n and bar.c = '@param0' \n and bar.b = (SELECT j from baz where k < 1000);
"""

_TEST_QUERY_4_SUBQUERY_PREDICATE_SUBPLAN = """
SELECT x, y, a, b, d_date  FROM  foo JOIN bar on x = b WHERE  a < 2  and bar.c = 'bravo' and bar.b  = (select j from baz where j > foo.x and j > bar.b order by k LIMIT 1);
"""

_TEST_QUERY_5_SUBQUERY_PREDICATE_SUBPLAN_JOIN = """
SELECT x, y, a, b, d_date  FROM  foo JOIN bar on x = b WHERE  a < 2  and bar.c = 'bravo' and bar.b  = (select j from baz join foo f1 on f1.x = baz.j where j > foo.x and j > bar.b order by k LIMIT 1);
"""

_BASE_HINTS = [{
    "hints": "hint0",
    "source": "source0"
}, {
    "hints": "hint1",
    "source": "source1"
}]

_TEST_DATA_DIR = "kepler/training_data_collection_pipeline/testdata"

# Typing aliases.
JSON = Any


def _plan_contains_node_with_parent_relationship(
    query_plan: JSON, parent_relationship: str) -> bool:

  if "Parent Relationship" in query_plan and query_plan[
      "Parent Relationship"] == parent_relationship:
    return True

  if "Plans" in query_plan:
    for child_plan in query_plan["Plans"]:
      has_target_parent_relationship = _plan_contains_node_with_parent_relationship(
          child_plan, parent_relationship)
      if has_target_parent_relationship:
        return True

  return False


class PgHintExtractorTest(absltest.TestCase):
  """These tests primarily rely on the following test data.

  Output of generate candidates using the following on test query 0:

    all_params = [['https://hello.com'],
                  ['http://goodbye.org/methods'],
                  ['http://www.goodnight.org'],
                  ['http://www.goodmorning.com']]
    configs = ['enable_hashjoin', 'enable_mergejoin']

    The default plan uses HashJoin. Hence effectively half the plans will
    use HashJoin, and half will not.
  """

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._database_configuration = query_utils.DatabaseConfiguration(
        dbname=self._test_database.dbname,
        user=test_util.USER,
        password=test_util.PASSWORD)
    self._query_manager = query_utils.QueryManager(self._database_configuration)
    test_util.populate_database(self._query_manager)

    plan_path = os.path.join(_TEST_DATA_DIR,
                             "generate_candidates_explain_plans.json")
    self.test_query_explain_plans = json.loads(
        resources.GetResource(plan_path))["output"][test_util.TEST_QUERY_ID]

  def tearDown(self):
    self._test_database.drop()
    super().tearDown()

  def test_basic_merge_hints(self):
    base_hints = copy.deepcopy(_BASE_HINTS)
    additional_hints = [{"hints": "hint2", "source": "source2"}]
    pg_plan_hint_extractor.merge_hints(base_hints, additional_hints)
    self.assertLen(base_hints, 3)
    for i in range(3):
      self.assertEqual(base_hints[i]["hints"], f"hint{i}")
      self.assertEqual(base_hints[i]["source"], f"source{i}")

  def test_merge_hints_identifier(self):
    base_hints = copy.deepcopy(_BASE_HINTS)
    additional_hints = [{"hints": "hint2", "source": "source2"}]
    merge_suffix = "_merge_stage1"
    pg_plan_hint_extractor.merge_hints(base_hints, additional_hints,
                                       merge_suffix)
    self.assertLen(base_hints, 3)
    self.assertEqual(base_hints[2]["source"], "source2" + merge_suffix)

  def test_nonunique_merged_hints(self):
    base_hints = copy.deepcopy(_BASE_HINTS)
    additional_hints = [{"hints": "hint0", "source": "source2"}]
    pg_plan_hint_extractor.merge_hints(base_hints, additional_hints)
    self.assertLen(base_hints, 2)
    for hint in base_hints:
      self.assertNotEqual(hint["source"], "source2")

  def test_extract_single_query_hint(self):
    hints = pg_plan_hint_extractor.get_single_query_hints(
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_0)

    expected = ("/*+  SeqScan(foo) SeqScan(bar) HashJoin(foo bar) "
                "Leading((foo bar)) */")
    self.assertEqual(hints, expected)

  def test_extract_hint_subquery_filter(self):
    explain_plan = self._query_manager.get_query_plan(
        _TEST_QUERY_3_SUBQUERY_PREDICATE_INITPLAN)["Plan"]
    self.assertTrue(
        _plan_contains_node_with_parent_relationship(explain_plan, "InitPlan"))
    hints = pg_plan_hint_extractor.get_single_query_hints(
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_3_SUBQUERY_PREDICATE_INITPLAN)
    expected = ("/*+  SeqScan(bar) SeqScan(foo) NestLoop(bar foo) "
                "Leading((bar foo)) SeqScan(baz) */")
    self.assertEqual(hints, expected)

    explain_plan = self._query_manager.get_query_plan(
        _TEST_QUERY_6_SUBQUERY_PREDICATE_INITPLAN_IS_SCAN_NODE)["Plan"]
    self.assertTrue(
        _plan_contains_node_with_parent_relationship(explain_plan, "InitPlan"))
    hints = pg_plan_hint_extractor.get_single_query_hints(
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_6_SUBQUERY_PREDICATE_INITPLAN_IS_SCAN_NODE)

    expected = ("/*+  SeqScan(bar) SeqScan(baz) */")
    self.assertEqual(hints, expected)

    explain_plan = self._query_manager.get_query_plan(
        _TEST_QUERY_4_SUBQUERY_PREDICATE_SUBPLAN)["Plan"]
    self.assertTrue(
        _plan_contains_node_with_parent_relationship(explain_plan, "SubPlan"))
    hints = pg_plan_hint_extractor.get_single_query_hints(
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_4_SUBQUERY_PREDICATE_SUBPLAN)
    expected = ("/*+  SeqScan(foo) SeqScan(bar) HashJoin(foo bar) "
                "Leading((foo bar)) SeqScan(baz) */")
    self.assertEqual(hints, expected)

    explain_plan = self._query_manager.get_query_plan(
        _TEST_QUERY_5_SUBQUERY_PREDICATE_SUBPLAN_JOIN)["Plan"]
    self.assertTrue(
        _plan_contains_node_with_parent_relationship(explain_plan, "SubPlan"))
    hints = pg_plan_hint_extractor.get_single_query_hints(
        database_configuration=self._database_configuration,
        query=_TEST_QUERY_5_SUBQUERY_PREDICATE_SUBPLAN_JOIN)

    expected = ("/*+  SeqScan(foo) SeqScan(bar) HashJoin(foo bar) "
                "Leading((foo bar)) SeqScan(f1) SeqScan(baz) "
                "HashJoin(f1 baz) Leading((f1 baz)) */")
    self.assertEqual(hints, expected)

    # Force a plan change in the subquery and verify hint extraction works.
    new_hint = ("/*+  SeqScan(foo) SeqScan(bar) HashJoin(foo bar) "
                "Leading((foo bar)) SeqScan(f1) SeqScan(baz) "
                "MergeJoin(f1 baz) Leading((baz f1)) */")
    new_query = new_hint + _TEST_QUERY_5_SUBQUERY_PREDICATE_SUBPLAN_JOIN
    explain_plan = self._query_manager.get_query_plan(new_query)["Plan"]
    self.assertTrue(
        _plan_contains_node_with_parent_relationship(explain_plan, "SubPlan"))
    hints = pg_plan_hint_extractor.get_single_query_hints(
        database_configuration=self._database_configuration, query=new_query)
    expected = ("/*+  SeqScan(foo) SeqScan(bar) HashJoin(foo bar) "
                "Leading((foo bar)) SeqScan(baz) SeqScan(f1) "
                "MergeJoin(baz f1) Leading((baz f1)) */")
    self.assertEqual(hints, expected)

  def test_nonunique_explain_plans(self):
    """Test if some explain plans are the same, then the hints should be the same.

    See the explain plan description for details about the generation process.
    In particular, the default plan uses hash join and we include
    enable_hashjoin as a flag, so half of them should have hashjoin disabled.
    """
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                self.test_query_explain_plans)
    counts, _, _, _ = plan_hint_extractor.get_consolidated_plan_hints()

    # Half have hashjoin enabled, half disabled.
    self.assertLen(counts, 2)
    for _, hint_data in counts.items():
      self.assertEqual(hint_data["count"], 8)

  def test_plan_index_sorting(self):
    """Check that the default plan index corresponds to the right plan."""
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                self.test_query_explain_plans)
    _, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )

    for params_plan_index in params_plan_indices:
      plan_index = params_plan_index["plan_index"]
      # Default plan index should be 1 for all params.
      self.assertEqual(plan_index, 1)
      # Default plan should use hash join for all params.
      self.assertIn("HashJoin", plan_hints[plan_index]["hints"])

  def test_nonuniform_defaults(self):
    """Test case where there are multiple default plans over params.

    Included explain plan JSON has only one default plan. Manually alter
    one to be different, and check that everything works as expected.
    """
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()

    self.test_query_explain_plans[0]["result"]["Plan"][
        "Node Type"] = "Merge Join"

    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                self.test_query_explain_plans)
    _, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )

    for i, params_plan_index in enumerate(params_plan_indices):
      plan_index = params_plan_index["plan_index"]
      self.assertEqual(plan_index, 2 if i == 0 else 1)
      self.assertIn("MergeJoin" if i == 0 else "HashJoin",
                    plan_hints[plan_index]["hints"])

  def test_verify_hints(self):
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                self.test_query_explain_plans)
    _, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )

    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=test_util.TEST_QUERY_ID,
        query=_TEST_QUERY_0,
        plan_hints=plan_hints,
        params_plan_indices=params_plan_indices,
        database_configuration=self._database_configuration)
    for num_failures in failure_counts[test_util.TEST_QUERY_ID].values():
      self.assertEqual(num_failures, 0)

  def test_verify_forced_hints(self):
    """Manually force a join type, and check that pg_hint_plan respects it."""
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                self.test_query_explain_plans)
    _, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )

    # Force all joins to be MergeJoin.
    for hint in plan_hints:
      hint["hints"] = hint["hints"].replace("NestLoop", "MergeJoin").replace(
          "HashJoin", "MergeJoin")

    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=test_util.TEST_QUERY_ID,
        query=_TEST_QUERY_0,
        plan_hints=plan_hints,
        params_plan_indices=params_plan_indices,
        database_configuration=self._database_configuration)

    for num_failures in failure_counts[test_util.TEST_QUERY_ID].values():
      self.assertEqual(num_failures, 0)

  def test_verify_wrong_hints(self):
    """Test that verification failure counts can be positive.

    Here, we manually force verification failures by making up fake hints.
    """
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                self.test_query_explain_plans)
    _, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )

    # Replace NestLoop with fake join method "SingleLoop".
    for hint in plan_hints:
      # Bypass assertion that only non-default plans fail verification.
      if hint["source"] != "default":
        hint["hints"] = hint["hints"].replace("NestLoop", "SingleLoop")

    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=test_util.TEST_QUERY_ID,
        query=_TEST_QUERY_0,
        plan_hints=plan_hints,
        params_plan_indices=params_plan_indices,
        database_configuration=self._database_configuration)

    for hint, num_failures in failure_counts[test_util.TEST_QUERY_ID].items():
      if "SingleLoop" in hint:
        self.assertEqual(num_failures, 4)

  def test_row_no_join(self):
    explain_plan = self._query_manager.get_query_plan(_TEST_QUERY_1)["Plan"]
    row_counts = pg_plan_hint_extractor.extract_row_counts(explain_plan)
    self.assertEmpty(row_counts)

  def test_row_counts_one_join(self):
    explain_plan = self._query_manager.get_query_plan(_TEST_QUERY_0)["Plan"]
    row_counts = pg_plan_hint_extractor.extract_row_counts(explain_plan)
    self.assertLen(row_counts, 1)
    self.assertEqual(row_counts["foo bar"], 11)

  def test_row_counts_two_joins(self):
    explain_plan = self._query_manager.get_query_plan(_TEST_QUERY_2)["Plan"]
    row_counts = pg_plan_hint_extractor.extract_row_counts(explain_plan)
    self.assertLen(row_counts, 2)
    self.assertEqual(row_counts["bar baz"], 1200)
    self.assertEqual(row_counts["foo bar baz"], 2712000)

  def test_repeated_call_rules(self):
    """Verify valid and invalid call orderings on PlanHintExtractor.

    Specifically, verify the ordering requirements of add_query_plans() and
    get_consolidated_plan_hints().
    """
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()

    counts, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )
    self.assertFalse(counts)
    self.assertFalse(plan_hints)
    self.assertFalse(params_plan_indices)

    # This call does not trigger a violation because the call to
    # get_consolidated_plan_hints() before any plans were added is logically a
    # no-op. This exception to the call ordering requirements seems harmless
    # because it does not affect correctness.
    plan_hint_extractor.add_query_plans(self.test_query_explain_plans[0])

    counts, plan_hints, params_plan_indices, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )
    self.assertTrue(counts)
    self.assertTrue(plan_hints)
    self.assertTrue(params_plan_indices)

    # Cannot call add_query_plan() again now that hints have been consolidated.
    self.assertRaisesRegex(ValueError, "Cannot call add_query_plan",
                           plan_hint_extractor.add_query_plans,
                           self.test_query_explain_plans[1])

    # Calling get_consolidated_plan_hints() again returns the same results.
    counts_again, plan_hints_again, params_plan_indices_again, _ = plan_hint_extractor.get_consolidated_plan_hints(
    )
    self.assertEqual(counts, counts_again)
    self.assertEqual(plan_hints, plan_hints_again)
    self.assertEqual(params_plan_indices, params_plan_indices_again)

if __name__ == "__main__":
  absltest.main()

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

"""Extract hints from an EXPLAIN plan into a simple format.

PlanHintExtractor takes plans generated using the EXPLAIN command (which are
tree-structured dicts) and extracts the hints (specifically scan methods, join
methods, and join order) into a string format that can be used to explicitly
run a hinted query. For each query template, produces a set of unique plans
extracted from all plans associated with all parameter values.

Also provides functionality to "verify hints" by re-extracting hints from
a hinted query, to check that the database optimizer uses the hints we specify
in the query.

Typical usage:
```
# queries: Dict containing query templates; is only used in verification.
hint_extractor = PlanHintExtractor()

# counts: Statistics about number of unique hints (plans) extracted per query.
# hints: Dict mapping query_id to info about extracted hints.
# param_indices: For each query_id, for each set of param values record
#   the index of the plan the default optimizer plan corresponds to.
# query_id: ID of query to extract
# plans_info: Explain plan content under "output" key.
counts, hints, param_indices = hint_extractor.extract_plan_hints(query_id,
plans_info)
failed_counts = verify_hints(query_id, query, hints, param_indices,
                             database_name)
```
"""

import copy
import functools
import json
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging

from kepler.training_data_collection_pipeline import query_utils

# key = Node Type in explain plan.
# value = Name used by pg_hint_plan.
_SCAN_METHOD_TRANSLATOR = {
    'Seq Scan': 'SeqScan',
    'Index Scan': 'IndexScan',
    'Index Only Scan': 'IndexOnlyScan',
    'Bitmap Heap Scan': 'BitmapScan'
}
_SCAN_METHOD_RED_HERRINGS = ['Subquery Scan', 'CTE Scan']
_JOIN_METHOD_TRANSLATOR = {
    'Hash Join': 'HashJoin',
    'Nested Loop': 'NestLoop',
    'Merge Join': 'MergeJoin'
}
_SUBQUERY_PARENT_RELATIONSHIPS = ['InitPlan', 'SubPlan']

# Typing aliases.
JSON = Any
Hint = Dict[str, str]
ParamsDefaultIndex = Dict[str, Union[List[str], int]]


def merge_hints(base: List[Hint],
                extra_hints: List[Hint],
                merge_suffix: str = '') -> None:
  """Append any hints in extra_hints to base if they are not already present.

  Only considers 'hint' key when testing for uniqueness.

  Args:
    base: List of base set of hints (each a dict containing hint and source)
    extra_hints: List of additional hints to merge.
    merge_suffix: Optional string to append to each merged hint's source.
  """
  existing_hints = set([hint['hints'] for hint in base])
  for extra_hint in extra_hints:
    if extra_hint['hints'] not in existing_hints:
      existing_hints.add(extra_hint['hints'])
      extra_hint['source'] += merge_suffix
      base.append(extra_hint)


def verify_hints(
    query_id: str, query: str, plan_hints: List[Hint],
    params_plan_indices: List[ParamsDefaultIndex],
    database_configuration: query_utils.DatabaseConfiguration
) -> Dict[str, Dict[str, int]]:
  """Verify that re-extraction produces the same set of hints.

  For each query_id and hint, perform re-extraction on hinted query, and
  verify that those hints are identical to the original hints.

  Args:
    query_id: ID of query to verify for.
    query: SQL query template string with 0 or more parameters provided in the
      form of @param#, starting with 0.
    plan_hints: See output of extract_plan_hints.
    params_plan_indices: See output of extract_plan_hints.
    database_configuration: The configuration describing the database
      connection.

  Returns:
    Dict mapping query_id to hint to number of failures.
  """
  logging.info('Checking %s hints for query_id %s', len(plan_hints), query_id)
  failure_counts = {}
  failure_counts[query_id] = {}

  for hint in plan_hints:
    curr_hinted_query = '{} {}'.format(hint['hints'], query)
    params_failure_count = 0

    with multiprocessing.Pool() as pool:
      # This usage of imap_unordered does not have externally visible
      # effects. The only output produced is a count.
      for result in pool.imap_unordered(
          functools.partial(_check_hint, database_configuration, hint['hints'],
                            curr_hinted_query),
          params_plan_indices,
          chunksize=max(10, min(100, int(len(params_plan_indices) / 100)))):
        if not result:
          params_failure_count += 1
    failure_counts[query_id][hint['hints']] = params_failure_count
    if params_failure_count:
      logging.info('Failure rate for hint %s source %s: %s/%s',
                   hint['hints'], hint['source'], params_failure_count,
                   len(params_plan_indices))
  return failure_counts


def _is_scan_node(node: JSON) -> bool:
  return node['Node Type'] in _SCAN_METHOD_TRANSLATOR


def _is_subquery_node(node: JSON, parent_relationship_required: bool) -> bool:
  if parent_relationship_required:
    assert 'Parent Relationship' in node
  return (('Parent Relationship' in node) and
          (node['Parent Relationship'] in _SUBQUERY_PARENT_RELATIONSHIPS))


def _strip_parens(text: str) -> str:
  return text.replace(')', '').replace('(', '')


def _extract_scan_method_hints(node: JSON, plan_hints: List[str]) -> None:
  """Extract scan methods from explain plan.

  Args:
    node: Dict corresponding to a (sub-)plan.
    plan_hints: List to populate with scan method hints.
  """
  if _is_subquery_node(node, parent_relationship_required=False):
    # Hints for subqueries are handled separately.
    return

  if _is_scan_node(node):
    if 'Index Name' in node:
      plan_hints.append('{}({} {})'.format(
          _SCAN_METHOD_TRANSLATOR[node['Node Type']], node['Alias'],
          node['Index Name']))
    else:
      plan_hints.append('{}({})'.format(
          _SCAN_METHOD_TRANSLATOR[node['Node Type']], node['Alias']))
  else:
    assert 'Scan' not in node['Node Type'] or node[
        'Node Type'] in _SCAN_METHOD_RED_HERRINGS, ('Unexpected Scan Node: {} '
                                                    'from {}').format(
                                                        node['Node Type'],
                                                        json.dumps(node))

    if 'Plans' in node:
      for child_node in node['Plans']:
        _extract_scan_method_hints(child_node, plan_hints)


def _extract_join_hints(node: JSON,
                        plan_hints: List[str],
                        row_counts: Optional[Dict[str, int]] = None) -> str:
  """Extract join methods from explain plan.

  Args:
    node: Dict corresponding to a (sub-)plan.
    plan_hints: List to populate with join method hints.
    row_counts: Dict to populate with estimated row counts.

  Returns:
    String corresponding to join order of plan under node, demarked
      by parentheses.
  """
  join_method = None
  if 'Join Type' in node:
    join_method = _JOIN_METHOD_TRANSLATOR[node['Node Type']]

  if not join_method:
    if _is_scan_node(node):
      return node['Alias']

    return _extract_join_hints(node['Plans'][0], plan_hints, row_counts)

  # Iterate the JOINs child nodes. We expect one with Parent Relationship
  # "Outer" and one with Parent Relationship "Inner" in that order. Any
  # additional nodes must be _is_subquery_node(). Hints for subqueries are
  # handled separately.
  join_child_index = 0
  while node['Plans'][join_child_index]['Parent Relationship'] != 'Outer':
    assert _is_subquery_node(
        node['Plans'][join_child_index], parent_relationship_required=True), (
            'Join with unexpected node while searching for Outer: {}'.format(
                json.dumps(node)))
    join_child_index += 1
  outer_part = _extract_join_hints(node['Plans'][join_child_index], plan_hints,
                                   row_counts)
  join_child_index += 1

  while node['Plans'][join_child_index]['Parent Relationship'] != 'Inner':
    assert _is_subquery_node(
        node['Plans'][join_child_index], parent_relationship_required=True), (
            'Join with unexpected node while searching for Inner: {}'.format(
                json.dumps(node)))
    join_child_index += 1
  inner_part = _extract_join_hints(node['Plans'][join_child_index], plan_hints,
                                   row_counts)
  join_child_index += 1

  while join_child_index < len(node['Plans']):
    assert _is_subquery_node(
        node['Plans'][join_child_index], parent_relationship_required=True
    ), ('Join with unexpected node after finding Outer and Inner: {}'.format(
        json.dumps(node)))
    join_child_index += 1

  join_order = '({} {})'.format(outer_part, inner_part)
  joined_tables = _strip_parens(join_order)
  plan_hints.append('{}({})'.format(join_method, joined_tables))
  if node.get('Plan Rows') and row_counts is not None:
    row_counts[joined_tables] = node.get('Plan Rows')

  return join_order


def extract_row_counts(explain_plan: JSON) -> Dict[str, int]:
  """Extracts row counts for joins only."""
  row_counts = {}
  _ = _extract_join_hints(explain_plan, [], row_counts)
  return row_counts


def extract_base_table_row_counts(explain_plan: JSON) -> Dict[str, int]:
  """Extracts row counts for base tables only."""
  row_counts = {}

  def _extract_base_counts_helper(node):
    if _is_scan_node(node):
      row_counts[node['Alias']] = node['Plan Rows']
    for child in node.get('Plans', []):
      _extract_base_counts_helper(child)

  _extract_base_counts_helper(explain_plan)

  return row_counts


def _extract_plan_hints_builder(explain_plan_subtree_root: JSON) -> str:
  """Computes string representation of hints for a given EXPLAIN plan subtree.

  Args:
    explain_plan_subtree_root: Dict containing the explain plan for a subtree of
      the full EXPLAIN plan.

  Returns:
    Space-joined string of the following hints in order: scan methods, join
    methods, join order. Subquery hints are appended after the main query's join
    order and following the same ordering for hint components.
  """
  plan_hints = []
  _extract_scan_method_hints(explain_plan_subtree_root, plan_hints)
  join_order = _extract_join_hints(explain_plan_subtree_root, plan_hints)
  if join_order.count(' '):
    # Only add a Leading hint if there are multiple tables.
    plan_hints.append('Leading({})'.format(join_order))

  subquery_hints = []
  _extract_plan_hints_subqueries(explain_plan_subtree_root, subquery_hints)
  plan_hints.extend(subquery_hints)
  return ' '.join(plan_hints)


def _extract_plan_hints_subqueries(node: JSON, plan_hints: List[str]) -> None:
  """Extracts the hints for any subqueries.

  Args:
    node: Dict corresponding to a (sub-)plan.
    plan_hints: List to populate with plan hints from subqueries.
  """
  if _is_subquery_node(node, parent_relationship_required=False):
    subtree_node = copy.deepcopy(node)
    del subtree_node['Parent Relationship']
    plan_hints.append(_extract_plan_hints_builder(subtree_node))
  else:
    if 'Plans' in node:
      for child_node in node['Plans']:
        _extract_plan_hints_subqueries(child_node, plan_hints)


# pg_hint_plan requires using the aliases for annotations, if they exist.
def _extract_plan_hints(explain_plan: JSON) -> str:
  """Computes string representation of hints for a given EXPLAIN plan.

  Args:
    explain_plan: Dict containing the explain plan (contents of top level key
      'Plan').

  Returns:
    Space-joined string of the following hints in order: scan methods, join
    methods, join order. Subquery hints are appended after the main query's join
    order and following the same ordering for hint components. The string
    includes starting and ending comment markers so that the hint can be used
    directly by prepending it to a query.
  """
  plan_hints = ['/*+ ']
  plan_hints.append(_extract_plan_hints_builder(explain_plan))
  plan_hints.append('*/')
  return ' '.join(plan_hints)


def get_file_content(filename: str) -> List[str]:
  """Read a single query template from a file.

  Args:
    filename: Filepath to file containing template for a single query.

  Returns:
    List of lines of parsed query.
  """
  lines = []
  with open(filename) as f:
    for line in f:
      if '--' in line:
        continue
      line = line.replace(';', '').strip()

      from_index = line.lower().find('from')
      if from_index > -1 and len(line) > 4 and (
          from_index == 0 or line[from_index - 1] == ' ') and (
              (len(line) == (from_index + 4)) or line[from_index + 4] == ' '):
        from_token = line[from_index:from_index + 4]
        components = line.split(from_token)
        assert len(components) == 2
        if components[0]:
          lines.append(components[0])
        lines.append(from_token)
        if components[1]:
          lines.append(components[1])
      else:
        lines.append(line)

  return lines


def get_single_query_hints(
    database_configuration: query_utils.DatabaseConfiguration,
    query: str,
    params: Optional[List[Any]] = None) -> str:
  """Extract the plan hints corresponding to a single query.

  Args:
    database_configuration: The configuration describing the database
      connection.
    query: The query in string format.
    params: List of parameters.

  Returns:
    String representation of the hints extracted from the explain plan.
  """
  query_manager = query_utils.QueryManager(database_configuration)
  return get_single_query_hints_with_plan(query_manager, query, params)[0]


def get_single_query_hints_with_plan(query_manager, query,
                                     params) -> Tuple[str, JSON]:
  explain_plan = query_manager.get_query_plan(query, params)
  return _extract_plan_hints(explain_plan['Plan']), explain_plan


def _get_plan_hint_index(plan_hints: List[Dict[str, str]],
                         plan_hint: str) -> int:
  for i, entry in enumerate(plan_hints):
    if entry['hints'] == plan_hint:
      return i

  assert False


def _generate_hints(
    params_plan_info: JSON) -> Tuple[List[str], List[str], List[str],
                                     List[JSON]]:
  """Extract hints for all plans for a single parameter value.

  Args:
   params_plan_info: Dict containing additional_plans, result (default plan),
     params.

  Returns:
    Tuple consisting of:
       1. List of param values.
       2. List of extracted hints for each plan.
       3. List of sources (default for un-hinted plan, and set of hint configs
       for the rest).
       4. List of debug_info dicts (empty if they don't exist).
  """
  hints = []
  sources = []

  hints.append(_extract_plan_hints(params_plan_info['result']['Plan']))
  sources.append('default')

  # Extract hints from the additional plans.
  if 'additional_plans' in params_plan_info:
    for additional_plan, source in zip(params_plan_info['additional_plans'],
                                       params_plan_info['sources']):
      additional_hints = _extract_plan_hints(additional_plan['Plan'])
      hints.append(additional_hints)
      sources.append(source)

  return (params_plan_info['params'], hints, sources,
          params_plan_info.get('debug_info', {}))


def _check_hint(database_configuration: query_utils.DatabaseConfiguration,
                old_hint: str, curr_hinted_query: str,
                params_default_index: ParamsDefaultIndex) -> bool:
  """Sanity check that the given hints are actually used by pg to make the plan.

  After extracting the hints to produce a hinted query, we compute the
  corresponding plan, then extract the hints from the new plan. The hints
  re-extracted from the hinted query's EXPLAIN plan should be identical the
  the originally-extracted hints.

  Args:
    database_configuration: The configuration describing the database
      connection.
    old_hint: Extracted hint string from original plan.
    curr_hinted_query: Query using old_hint.
    params_default_index: Dict containing 'params' mapping to single set of
      param values.

  Returns:
    Boolean corresponding to whether the new and original hints are equal.
  """
  query_manager = query_utils.QueryManager(database_configuration)
  new_plan = query_manager.get_query_plan(curr_hinted_query,
                                          params_default_index['params'])
  new_hint = _extract_plan_hints(new_plan['Plan'])
  if old_hint != new_hint:
    logging.error('\nOld_Hint: %s \nNew_Hint: %s\nQuery: %s\nParams: %s',
                  old_hint, new_hint, curr_hinted_query,
                  str(params_default_index['params']))
    return False
  return True


class PlanHintExtractor:
  """Class for extracting hints from explain plan outputs into string format.

  This output can then be directly used, in the case of pg_hint_plan, to run
  a hinted query.

  Each instance of a PlanHintExtractor is intended to be used for a single query
  template. This invariant is not explicitly enforced and is the responsibility
  of the caller.

  Usage:
    1. Add as many query plans as desired for a single query_id via
      add_query_plans().
    2. Call get_consolidated_plan_hints() to retrieve hints after all
      query plans are added. After get_consolidated_plan_hints() has been
      called, it is an error to call add_query_plans() again. Repeated calls
      to get_consolidated_plan_hints() will yield the same results.
  """

  def __init__(self):
    self._counts = {}
    self._plan_hints = []
    self._params_plan_indices = []
    self._debug_infos = []

  def add_query_plans(self, params_plans_info: JSON) -> None:
    """Performs hint extraction on the provided query plans.

    Extracts hints for all the query plans in plans_info 'results' and
    'additional_plans'.  Accumulates hints into internal state for
    deduplication and indexing.

    The provided query plans must all be for a single instance of parameter bind
    values.

    Args:
      params_plans_info: A dict containing 'params', 'result', 'additional
        plans' and 'sources', as defined in pg_generate_plan_candidates
        functions such as get_query_plan.
    """
    if self._plan_hints:
      raise ValueError('Cannot call add_query_plan() after calling '
                       'get_consolidated_plan_hints()')

    params, hints, sources, debug_infos = _generate_hints(params_plans_info)

    if not hints:
      logging.info('Failed to extract hints for %s', params_plans_info)
      return

    for hint, source in zip(hints, sources):
      if hint not in self._counts:
        self._counts[hint] = {'count': 0, 'source': source}
      self._counts[hint]['count'] += 1

    # Plan indexes are set during get_consolidated_plan_hints. We store the plan
    # hints as a proxy key for now.
    params_default_index = {'params': params, 'plan_index': hints[0]}
    self._params_plan_indices.append(params_default_index)
    self._debug_infos.append(debug_infos)

  def get_num_hints(self) -> int:
    """Returns the current number of unique hints."""
    return len(self._counts)

  def get_consolidated_plan_hints(
      self
  ) -> Tuple[Dict[str, Dict[str, Union[int, str]]], List[Hint],
             List[JSON], List[JSON]]:
    """Returns deduplicated and indexed plan hints for the query.

    This function should be called to retrieve the consolidated results after n
    calls to add_query_plan(). Calling the function repeatedly will return the
    same values.

    Returns:
      Tuple consisting of the following:
        1. counts: Dict containing number of unique
           plans (hints) generated, and the first source that generated it
        2. plan_hints: list of unique hints (in lexicographical order).
           Each hint is a dict containing the hint string and the source.
        3. params_default_index: List containing dict mapping 'params' to
           list of parameters, and 'plan_index' to the index of the default
           plan.
        4. debug_infos: List of dicts containing debug info for each param.
    """
    if self._plan_hints:
      # This function has already been called before. Return precomputed
      # results.
      return (self._counts, self._plan_hints, self._params_plan_indices,
              self._debug_infos)

    for plan_hint in sorted(self._counts):
      self._plan_hints.append({
          'hints': plan_hint,
          'source': self._counts[plan_hint]['source']
      })

    for params_default_index in self._params_plan_indices:
      params_default_index['plan_index'] = _get_plan_hint_index(
          self._plan_hints, params_default_index['plan_index'])

    return (self._counts, self._plan_hints, self._params_plan_indices,
            self._debug_infos)


def add_query_plans_bulk(plan_hint_extractor: PlanHintExtractor,
                         params_plans_info_list: List[JSON]) -> None:
  """Adds params_plans_info for many parameters to plan_hint_extractor.

  This helper captures a repeated usage pattern in a shared function.

  Args:
    plan_hint_extractor: The PlanHintExtractor to which to add query plans per
      parameter.
    params_plans_info_list: A list of params_plans_info where each element
      corresponds to a unique set of parameter bind values for the same query
      template. The format of params_plans_info is described in
      PlanHintExtractor.add_query_plans().
  """
  for params_plans_info in params_plans_info_list:
    plan_hint_extractor.add_query_plans(params_plans_info)

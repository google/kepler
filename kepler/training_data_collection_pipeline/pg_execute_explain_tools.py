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

"""A suite of tools for efficiently collecting EXPLAIN plans.

Each of the tools here collects EXPLAIN plans in some fashion to support
analyses of Kepler ideas and baselines.

As different from pg_execute_training_data_queries which is single-threaded to
execute queries in isolation, this tool collects EXPLAIN plans in parallel.

"""
import enum
import functools
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import utils

# Typing aliases.
JSON = Any

# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = "####"

# The multiplier by which to scale up (down) the default plan cardinality
# estimate. We then binary search between the default plan cardinality estimate
# and the scaled value to find which cardinality estimate causes a plan
# change. The multiplier provides slightly more flexibility than setting a fixed
# max possible estimate.
_CARDINALITY_MULTIPLIER = 1 << 27

# The distributed_query_manager is initialized per-process to open a database
# connection exactly once per process when using a multiprocessing pool to
# parallelize _generate_explain_plan.
distributed_query_manager = None


def _init_per_process_global_query_manager(
    database_configuration: query_utils.DatabaseConfiguration):
  global distributed_query_manager
  distributed_query_manager = query_utils.QueryManager(database_configuration)


def _get_row_hinted_query(query: str, tables: str, row_count: int) -> str:
  return f"/*+ Rows({tables} #{row_count}) */ {query}"


def _does_plan_change(query_manager: query_utils.QueryManager,
                      base_query_plan_hints: str, query: str, params: List[Any],
                      tables: str, proposed_row_count: int) -> bool:
  hints, _ = pg_plan_hint_extractor.get_single_query_hints_with_plan(
      query_manager=query_manager,
      query=_get_row_hinted_query(
          query=query, tables=tables, row_count=proposed_row_count),
      params=params)
  return hints != base_query_plan_hints


def _find_plan_changing_cardinality(query_manager: query_utils.QueryManager,
                                    base_query_plan_hints: str, query: str,
                                    params: List[Any], tables: str,
                                    search_direction_is_increasing: bool,
                                    row_count_min: int,
                                    row_count_max: int) -> Optional[int]:
  """Identify the smallest cardinality change that causes a plan change.

  Args:
    query_manager: The QueryManager to provide database access.
    base_query_plan_hints: The starting set of hints used as a basis to
      determine whether the plan has changed. This is typically the hints for
      the default plan.
    query: Query in string format.
    params: List of parameter binding values.
    tables: The tables involved in the JOIN whose cardinality is being altered.
    search_direction_is_increasing: A boolean describing whether the
      base_query_plan_hints plan's cardinality is provided as row_count_min or
      row_count_max. If base_query_plan_hints represents row_count_min, that
      means we will search for the smallest *increase* in cardinality that
      causes a plan change.
    row_count_min: The lower bound for this iteration of binary search.
    row_count_max: The upper bound for this iteration of binary search.

  Returns:
    The smallest cardinality change that causes a plan change. None if the plan
    never changes.
  """
  # Round proposed_row_count up when searching in the decreasing direction.
  proposed_row_count = int(
      max((row_count_min + row_count_max +
           (not search_direction_is_increasing)) // 2, 1))

  hints_match = not _does_plan_change(
      query_manager=query_manager,
      base_query_plan_hints=base_query_plan_hints,
      query=query,
      params=params,
      tables=tables,
      proposed_row_count=proposed_row_count)

  if row_count_min == row_count_max:
    return proposed_row_count if not hints_match else None

  # We assume the hints will stay in the same until the change is large (small)
  # enough. We also assume the hints once changed, will not match
  # base_query_plan_hints again. Therefore, when search_direction_is_increasing,
  # search a higher range of values if the hints match and search the lower
  # range if the hints are already different. Do the opposite if
  # !search_direction_is_increasing, meaning our initial max corresponds to
  # base_query_plan_hints and we are looking for a change as the hints decrease.
  if (hints_match and
      search_direction_is_increasing) or (not hints_match and
                                          not search_direction_is_increasing):
    new_row_count_min = proposed_row_count + search_direction_is_increasing
    new_row_count_max = row_count_max
  else:
    new_row_count_min = row_count_min
    new_row_count_max = proposed_row_count - (
        not search_direction_is_increasing)

  return _find_plan_changing_cardinality(
      query_manager=query_manager,
      base_query_plan_hints=base_query_plan_hints,
      query=query,
      params=params,
      tables=tables,
      search_direction_is_increasing=search_direction_is_increasing,
      row_count_min=new_row_count_min,
      row_count_max=new_row_count_max)


def _get_table_count(join_tables: str) -> int:
  return join_tables.count(" ") + 1


def _calculate_plan_changing_cardinality_estimate_helper(
    query: str, params: List[Any]) -> Tuple[str, JSON]:
  """Calculates the cardinalities that effect a plan change from the default.

  For each join, we compute the cardinality estimate for that join that causes
  the plan to change away from the default plan, if one exists. The search
  checks both increasing the cardinality estimate as well as reducing it.

  Args:
    query: Query in string format.
    params: List of parameter binding values.

  Returns:
    A tuple containing:
      1) The params argument this was called with.
      2) A dict describing the cardinality estimate, joined tables, and
        alternative cardinality estimate required to change away from the
        default plan for params.
  """
  params_results = {}

  hints, explain_plan = pg_plan_hint_extractor.get_single_query_hints_with_plan(
      query_manager=distributed_query_manager, query=query, params=params)
  row_counts_by_joined_tables = pg_plan_hint_extractor.extract_row_counts(
      explain_plan["Plan"])
  max_table_count = None
  if row_counts_by_joined_tables:
    max_table_count = max(map(_get_table_count, row_counts_by_joined_tables))

  for join_tables, row_count in row_counts_by_joined_tables.items():
    table_count = _get_table_count(join_tables)
    if table_count == max_table_count:
      # Hinting the final JOIN will not change any of the optimizer decisions we
      # work with.
      continue

    plan_tipping_point_increase = None
    proposed_row_count_max = row_count * _CARDINALITY_MULTIPLIER
    # Short-circuit the binary search if we know the plan never changes.
    if _does_plan_change(
        query_manager=distributed_query_manager,
        base_query_plan_hints=hints,
        query=query,
        params=params,
        tables=join_tables,
        proposed_row_count=proposed_row_count_max):
      plan_tipping_point_increase = _find_plan_changing_cardinality(
          query_manager=distributed_query_manager,
          base_query_plan_hints=hints,
          query=query,
          params=params,
          tables=join_tables,
          search_direction_is_increasing=True,
          row_count_min=row_count,
          row_count_max=proposed_row_count_max)
      assert plan_tipping_point_increase is not None
    plan_tipping_point_decrease = None
    proposed_row_count_min = max(1, row_count / _CARDINALITY_MULTIPLIER)
    if _does_plan_change(
        query_manager=distributed_query_manager,
        base_query_plan_hints=hints,
        query=query,
        params=params,
        tables=join_tables,
        proposed_row_count=proposed_row_count_min):
      plan_tipping_point_decrease = _find_plan_changing_cardinality(
          query_manager=distributed_query_manager,
          base_query_plan_hints=hints,
          query=query,
          params=params,
          tables=join_tables,
          search_direction_is_increasing=False,
          row_count_min=proposed_row_count_min,
          row_count_max=row_count)
      assert plan_tipping_point_decrease is not None

    params_results[table_count] = {
        "row_count": row_count,
        "join_tables": join_tables,
        "plan_tipping_point_increase": plan_tipping_point_increase,
        "plan_tipping_point_decrease": plan_tipping_point_decrease
    }

  params_key = _NAME_DELIMITER.join([str(p) for p in params])
  return params_key, params_results


@enum.unique
class ExplainExtractionFunction(enum.Enum):
  TOTAL_COSTS = "total_costs"
  ESTIMATED_CARDINALITIES = "estimated_cardinalities"
  COMBINED = "combined"


def _extract_total_cost(query_plan: JSON) -> JSON:
  return [{"total_cost": query_plan["Plan"]["Total Cost"]}]


def _extract_cardinality_estimates(query_plan: JSON) -> JSON:
  row_counts = pg_plan_hint_extractor.extract_base_table_row_counts(
      query_plan["Plan"])
  return [{"estimated_cardinalities": row_counts}]


def _extract_combined(query_plan: JSON) -> JSON:
  res = _extract_total_cost(query_plan)
  res[0].update(_extract_cardinality_estimates(query_plan)[0])
  return res


def _collect_explain_plans_helper(
    database_configuration: query_utils.DatabaseConfiguration, query: str,
    plan_hints: List[Dict[str, str]], params: List[str],
    extract_function: ExplainExtractionFunction) -> Tuple[str, JSON]:
  """Extracts information from the EXPLAIN for a parameter and all query plans.

  Args:
    database_configuration: Used to create QueryManager for each parameter.
    query: Query in string format.
    plan_hints: Hints representing all the query plans for which to collect
      EXPLAINs.
    params: List of parameter binding values.
    extract_function: Specifies what information to extract from the EXPLAIN.

  Returns:
    A tuple containing:
      1) The params argument this was called with.
      2) A dict containing the extract EXPLAIN plan info for each query
        plan. This dict uses the same structure as
        pg_execute_training_data_queries.
  """
  query_manager = query_utils.QueryManager(database_configuration)
  params_results = {}
  params_results["results"] = []
  for hint in plan_hints:
    hinted_query = utils.get_hinted_query(
        query=query, hints=hint["hints"])
    query_plan = query_manager.get_query_plan(hinted_query, params)
    if extract_function == ExplainExtractionFunction.TOTAL_COSTS:
      params_results["results"].append(_extract_total_cost(query_plan))
    elif extract_function == ExplainExtractionFunction.ESTIMATED_CARDINALITIES:
      params_results["results"].append(
          _extract_cardinality_estimates(query_plan))
    elif extract_function == ExplainExtractionFunction.COMBINED:
      params_results["results"].append(_extract_combined(query_plan))

  params_key = _NAME_DELIMITER.join([str(p) for p in params])
  return params_key, params_results


def calculate_plan_changing_cardinality_estimates(
    database_configuration: query_utils.DatabaseConfiguration,
    query_id: str,
    templates: Any,
    parameter_values: Any,
    limit: Optional[int] = None,
    multiprocessing_chunksize: int = 100) -> JSON:
  """Calculates the cardinality estimate to cause a plan change.

  The purpose of this output is to evaluate the design and efficacy of Row Count
  Evolution.

  For each parameter and for each join, this function calculates the cardinality
  estimate required to effect a change in the query plan away from the default.

  Args:
    database_configuration: The configuration describing the database
      connection.
    query_id: The query id for which to execute queries.
    templates: A mapping from query id to the templatized SQL query text.
    parameter_values: A mapping from query id to all the parameter values to
      execute with for that query id.
    limit: The number of parameter_values to gather execution data for.
    multiprocessing_chunksize: The chunksize used when splitting work in imap.
      This is primarily exposed to ensure tests can run with multiple processes
      by setting a low multiprocessing_chunksize.

  Returns:
    A dict describing the cardinality estimate, joined tables, and alternative
    cardinality estimate required to change away from the default plan for each
    parameter binding set.
  """

  query = templates[query_id]["query"]

  results_by_parameter = {}

  with multiprocessing.Pool(
      initializer=_init_per_process_global_query_manager,
      initargs=[database_configuration]) as pool:
    for params_key, params_results in pool.imap(
        func=functools.partial(
            _calculate_plan_changing_cardinality_estimate_helper, query),
        iterable=parameter_values[query_id]["params"][:limit],
        chunksize=multiprocessing_chunksize):
      results_by_parameter[params_key] = params_results

  return {query_id: results_by_parameter}


def collect_explain_plan_info(
    database_configuration: query_utils.DatabaseConfiguration,
    query_id: str,
    templates: Any,
    parameter_values: Any,
    plan_hints: Any,
    extract_function: ExplainExtractionFunction,
    limit: Optional[int] = None,
    multiprocessing_chunksize: int = 100) -> JSON:
  """Collects the EXPLAIN plan info across parameters and query plans.

  For each set of parameter values and each query plan hint, we execute an
  EXPLAIN, extract the desired information, and store it into a data structure
  matching the one produced by pg_execute_training_data_queries.

  Since EXPLAIN is deterministic, this implementation uses multiprocessing to
  speed up the execution of EXPLAIN plans.

  Args:
    database_configuration: The configuration describing the database
      connection.
    query_id: The query id for which to execute queries.
    templates: A mapping from query id to the templatized SQL query text.
    parameter_values: A mapping from query id to all the parameter values to
      execute with for that query id.
    plan_hints: A mapping from query id to pg_hint_plan hints representing the
      set of query plans for execution.
    extract_function: Specifies what information to extract from the EXPLAIN.
    limit: The number of parameter_values to execute EXPLAINs for.
    multiprocessing_chunksize: The chunksize used when splitting work in imap.
      This is primarily exposed to ensure tests can run with multiple processes
      by setting a low multiprocessing_chunksize.

  Returns:
    A dict containing the extracted ifnormation of the explain plan for each
    query plan for each parameter. This dict uses the same structure as
    pg_execute_training_data_queries.
  """
  results_by_parameter = {}
  with multiprocessing.Pool(
      initializer=_init_per_process_global_query_manager,
      initargs=[database_configuration]) as pool:
    for params_key, params_results in pool.imap(
        func=functools.partial(
            _collect_explain_plans_helper,
            database_configuration,
            templates[query_id]["query"],
            plan_hints[query_id],
            extract_function=extract_function),
        iterable=parameter_values[query_id]["params"][:limit],
        chunksize=multiprocessing_chunksize):
      results_by_parameter[params_key] = params_results

  return {query_id: results_by_parameter}

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

"""Generates EXPLAIN plans across hints, cardinality estimates, and parameters.

Sample use-cases:
  1. Generate training data for Kepler GNN feature training after compiling the
     work request across a series of input files defining query templates,
     parameter values, and query plan hints.
  2. Investigate how much perturbing cardinality estimates exhaustively allows
     us to explore the plan space.
"""

import dataclasses
import functools
import itertools
import multiprocessing
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kepler.training_data_collection_pipeline import pg_execute_training_data_queries
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_plan_utils
from kepler.training_data_collection_pipeline import query_utils

# Typing aliases.
JSON = Any

_ROWS_HINT = 'Rows({} #{})'


@dataclasses.dataclass
class JoinEstimateDescription:
  """Describes a join and its output cardinality estimate.

  The tables string is in the format understood by the pg_hint_plan Rows hint.
  """
  tables: str
  cardinality_estimate: int


def _generate_rows_hint(row_counts_by_joined_tables: Dict[str, int],
                        cardinality_multipliers: Tuple[float]) -> str:
  """Generates Rows hints to force the updated cardinality estimates.

  Applies the cardinality_multipliers to the corresponding elements of
  row_counts_by_joined_tables and then composes a hint for join output
  cardinality estimates using pg_hint_plan syntax. The pg_hint_plan extension
  requires all forced estimates to be at least one row.

  Args:
    row_counts_by_joined_tables: Dict of row counts keyed by set of tables in
      the full join subtree as required by pg_hint_plan hint format.
    cardinality_multipliers: Multipliers to adjust the cardinality estimate for
      each join in row_counts_by_joined_tables.

  Returns:
    A string containing hints to force the updated join output cardinality
    estimates.
  """

  hints = []
  for (tables, row_count), cardinality_multiplier in zip(
      row_counts_by_joined_tables.items(), cardinality_multipliers):
    hints.append(
        _ROWS_HINT.format(tables, max(1,
                                      int(row_count * cardinality_multiplier))))

  return ' '.join(hints)


def _extract_query_hints(query: str) -> str:
  """Retrieves the pg_hint_plan hint section from the query string."""
  return query[:query.index('*/') + 2]


def _append_hint(query: str, additional_hint: str) -> str:
  """Appends the provided hint to the query hint section."""
  # We assume the only occurrence of */ is related to the hint comment. This
  # code will need to be amended if we encounter examples where "*/" appears
  # elsewhere in the query text.
  assert query.count('*/') == 1
  return query.replace('*/', additional_hint + ' */')


# The distributed_query_manager is initialized per-process to open a database
# connection exactly once per process when using a multiprocessing pool to
# parallelize _generate_explain_plan.
distributed_query_manager = None


def _init_per_process_global_query_manager(
    database_configuration: query_utils.DatabaseConfiguration):
  global distributed_query_manager
  distributed_query_manager = query_utils.QueryManager(database_configuration)


def _generate_explain_plan(query: str, params: List[Any],
                           row_counts_by_joined_tables: Dict[str, int],
                           verify_hints_unchanged: bool,
                           cardinality_multipliers: Tuple[float]) -> JSON:
  """Generates an explain plan produced after altering cardinality estimates.

  Args:
    query: The hinted query template for which to generate EXPLAIN plans.
    params: The parameter bindings for a single instance of the query.
    row_counts_by_joined_tables: Dict of row counts keyed by set of tables in
      the full join subtree as required by pg_hint_plan hint format.
    verify_hints_unchanged: Set true if the extracted plan hints from the plan
      generated with perturbed cardinalities should be required to match the
      original plan hints.
      Example use-cases:
        1) Set true when perturbing cardinality to observe effects on cost when
           join and scan decisions are fixed.
        2) Set false when perturbing cardinality to observe the variety of plans
           generated based on the interaction of cardinality estimates and the
           database cost model.
    cardinality_multipliers: Multipliers to adjust the cardinality estimate for
      each join in row_counts_by_joined_tables.

  Returns:
    The EXPLAIN produced by adjusting cardinality estimates by
    cardinality_multipliers.
  """
  rows_hint = _generate_rows_hint(row_counts_by_joined_tables,
                                  cardinality_multipliers)
  updated_query = _append_hint(query, rows_hint)

  # Verify the following:
  # 1) The rows hint was incorporated into EXPLAIN.
  # 2) The plan hints extracted from the modified EXPLAIN still match the
  # original plan hints, confirming that the plan was not changed in terms
  # of scans and joins.
  verification_query_hints, verification_explain_plan = pg_plan_hint_extractor.get_single_query_hints_with_plan(
      distributed_query_manager, updated_query, params)

  if verify_hints_unchanged:
    query_hints = _extract_query_hints(query)
    assert query_hints == verification_query_hints, (
        f'Query hints changed. Got \n{verification_query_hints}\n '
        f'instead of \n{query_hints}')

  return verification_explain_plan


def multiplicatively_perturb_plan_cardinalities(
    query_manager: query_utils.QueryManager,
    query_id: str,
    templates: Any,
    parameter_values: Any,
    plan_hints: Any,
    cardinality_multipliers: List[float],
    verify_hints_unchanged: bool = True,
    limit: Optional[int] = 1,
    keys_to_remove: Optional[Sequence[str]] = None,
    multiprocessing_chunksize: int = 100) -> Any:
  """Generates EXPLAIN plans across hints, cardinalities, and parameters.

  For each hint, we retrieve query plans across a range of hypothetical
  cardinalities per JOIN. The routine is carried out with a requested number
  of parameters.

  Args:
    query_manager: The QueryManager to provide database access.
    query_id: The query id for which to execute queries.
    templates: A mapping from query id to the templatized SQL query text.
    parameter_values: A mapping from query id to all the parameter values to
      execute with for that query id.
    plan_hints: A mapping from query id to pg_hint_plan hints representing the
      set of query plans for execution.
    cardinality_multipliers: Multipliers to adjust the cardinality estimate for
      each join in row_counts_by_joined_tables.
    verify_hints_unchanged: Set true if the extracted plan hints from the plan
      generated with perturbed cardinalities should be required to match the
      original plan hints.
    limit: The number of parameter_values to gather execution data for.
    keys_to_remove: Keys to remove from each plan.
    multiprocessing_chunksize: The chunksize used when splitting work in
      imap. This is primarily exposed to ensure tests can run with multiple
      processes by setting a low multiprocessing_chunksize.

  Returns:
    EXPLAIN plans for each plan hint of query_id across the requested series for
    cardinality estimate overrides.
  """

  def _multiplicatively_perturb_plan_cardinalities_helper(
      unused_query_manager: query_utils.QueryManager,
      query: str,
      params: List[Any],
      _: Optional[int] = None,
  ) -> Tuple[List[JSON], None]:
    """Generates EXPLAIN plans across a series of cardinality perturbations.

    This function is passed through as the execute_query_fn argument to
    pg_execute_training_data_queries.execute_training_data_queries in the
    implementation of multiplicatively_perturb_plan_cardinalities. The third
    argument, an
    optional timeout in execute_query_fn, is not relevant to the usage here.

    Args:
      unused_query_manager: Unused; only for compatibility purposes.
      query: The hinted query template for which to generate EXPLAIN plans.
      params: The parameter bindings for a single instance of the query.

    Returns:
      Tuple:
        1) A list of EXPLAIN plans, each corresponding to a perturbation of
           cardinality_multipliers altering the row estimate for each JOIN
           output in the hinted query.
        2) None as the number of rows produced by this query.
    """
    del unused_query_manager  # External query manager used instead.
    explain_plans = []

    explain_plan = query_manager.get_query_plan(query, params)['Plan']
    row_counts_by_joined_tables = pg_plan_hint_extractor.extract_row_counts(
        explain_plan)

    # Generate an EXPLAIN plan for each element of the cross-product created
    # by multiplying each JoinEstimateDescription.cardinality by each element
    # of cardinality_multipliers.
    with multiprocessing.Pool(
        initializer=_init_per_process_global_query_manager,
        initargs=[query_manager.database_configuration]) as pool:
      for explain_plan in pool.imap(
          func=functools.partial(_generate_explain_plan, query, params,
                                 row_counts_by_joined_tables,
                                 verify_hints_unchanged),
          iterable=itertools.product(*([cardinality_multipliers] *
                                       len(row_counts_by_joined_tables))),
          chunksize=multiprocessing_chunksize):
        explain_plans.append(
            query_plan_utils.filter_keys(explain_plan, keys_to_remove))
    return explain_plans, None

  plans = {}

  def checkpoint_results(_: str, results: Any, is_metadata: bool) -> None:
    if is_metadata:
      return

    nonlocal plans
    plans = results

  pg_execute_training_data_queries.execute_training_data_queries(
      batch_index=0,
      parameter_values=parameter_values,
      query_id=query_id,
      templates=templates,
      plan_hints=plan_hints,
      iterations=1,  # 1 iteration sufficient for deterministic computation.
      batch_size=1000000,  # Checkpointing is cheap and not necessary.
      skip_indices=[],
      query_timeout_multiplier=1,  # Ignored by the provided execute_query_fn.
      query_timeout_min_ms=1,  # Ignored by the provided execute_query_fn.
      query_timeout_max_ms=1,  # Ignored by the provided execute_query_fn.
      execute_query_fn=_multiplicatively_perturb_plan_cardinalities_helper,
      checkpoint_results_fn=checkpoint_results,
      results_key='explain_output_across_cardinality',
      limit=limit,
  )

  return plans

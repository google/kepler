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

"""Executes SQL queries across parameter bindings and query plans.

This module provides as utility function used to generate training data for
Kepler after compiling the work request across a series of input files defining
query templates, parameter values, and query plan hints.

"""

import collections
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from absl import logging
import numpy as np

from kepler.data_management import database_simulator
from kepler.training_data_collection_pipeline import utils

# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = "####"

JSON = Any

PLAN_COVER = "plan_cover"


def _has_timeout(execution_results: List[JSON]) -> bool:
  return any(["timed_out" in result for result in execution_results])


def _get_plan_to_near_optimal_params(
    results: JSON,
    results_key: str,
    estimator: database_simulator.LatencyEstimator = database_simulator
    .LatencyEstimator.MIN,
    near_optimal_threshold: float = 1.01,
    num_params_limit: Optional[int] = None) -> Dict[int, Set[int]]:
  """Computes a mapping from plan index to params for which it is near optimal.

  Args:
    results: JSON mapping parameters to query execution results.
    results_key: A string corresponding to the key that maps to the result data.
    estimator: Specifies how to compute the estimated runtime from a list of
      latencies from repeated executions.
    near_optimal_threshold: If latency < this value * optimal latency, we say
      that this plan is near-optimal.
    num_params_limit: Use only the first N parameters.

  Returns:
    A defaultdict mapping the plan index to the set of parameter indices it is
      near-optimal for.
  """

  def _plan_latencies_checks(plan_latencies: Optional[List[JSON]]) -> bool:
    return (plan_latencies is not None and not _has_timeout(plan_latencies))

  estimator_func = database_simulator.ESTIMATOR_MAP[estimator]
  plan_to_params = collections.defaultdict(set)
  for i, results_entry in enumerate(results.values()):
    if num_params_limit is not None and i >= num_params_limit:
      break
    latencies = results_entry["results"]
    min_latencies = np.array([
        estimator_func([d.get(results_key, np.inf)
                        for d in t]) if _plan_latencies_checks(t) else np.inf
        for t in latencies
    ])
    optimal_latency = np.min(min_latencies)
    assert optimal_latency < np.inf

    near_optimal_idxs = np.where(
        min_latencies < optimal_latency * near_optimal_threshold)[0]
    for plan_index in near_optimal_idxs:
      plan_to_params[plan_index].add(i)

  return plan_to_params


def get_greedy_plan_cover(
    results: JSON,
    results_key: str,
    estimator: database_simulator.LatencyEstimator = database_simulator
    .LatencyEstimator.MIN,
    near_optimal_threshold: float = 1.01,
    num_params_threshold: float = 0.99,
    num_params_limit: Optional[int] = None) -> JSON:
  """Computes a near-optimal plan set cover using a greedy algorithm.

  Plan pruning poses the question: given a set of execution results for some
  number of initial plans, how can one select a subset of those plans such that
  each parameter still has a near-optimal plan? This can naturally be formulated
  as a set cover problem, which we relax here by only requiring that
  num_params_threshold of the parameters be covered, trading off coverage
  against the resulting cover size.

  This function implements the standard greedy approximation for the set cover,
  which iteratively adds the plan that is near-optimal for the greatest number
  of the remaining uncovered parameters.

  Some possible extensions include weighted set cover (e.g. by default latency),
  using a different estimator than min for warm-cache iterations, etc.

  Args:
    results: JSON mapping parameters to query execution results.
    results_key: A string corresponding to the key that maps to the result data.
    estimator: Specifies how to compute the estimated runtime from a list of
      latencies from repeated executions.
    near_optimal_threshold: If latency < this value * optimal latency, we say
      that this plan is near-optimal.
    num_params_threshold: Only attempt to cover this proportion of the parameter
      set.
    num_params_limit: Use only the first N parameters.

  Returns:
    Dict mapping chosen plan indices to the number of additional near-optimal
      params they cover.
  """
  plan_to_params = _get_plan_to_near_optimal_params(results, results_key,
                                                    estimator,
                                                    near_optimal_threshold,
                                                    num_params_limit)

  plan_cover = {}
  num_limit = len(results)
  if num_params_limit is not None:
    num_limit = min(num_limit, num_params_limit)
  uncovered_params = set(np.arange(num_limit))
  while uncovered_params:
    max_coverage = 0
    plan_index_to_add = None
    for plan_index, uncovered_params_for_plan in plan_to_params.items():
      coverage = len(uncovered_params_for_plan)
      if coverage > max_coverage:
        max_coverage = coverage
        plan_index_to_add = plan_index

    plan_cover[plan_index_to_add] = max_coverage
    params_to_remove = plan_to_params[plan_index_to_add].copy()
    uncovered_params.difference_update(params_to_remove)

    if len(uncovered_params) < (1 - num_params_threshold) * num_limit:
      break

    # Remove params that were covered in this iteration.
    for plan_uncovered_params in plan_to_params.values():
      plan_uncovered_params.difference_update(params_to_remove)

  return plan_cover


class PlanExecutionOrderManager:
  """Recommends plan execution order for the current parameter.

  The manager collects execution latencies for each plan executed so far and
  provides a recommended execution order based on all the execution latencies
  observed thus far.

  The recommended order sorts the plans from fastest to slowest based on prior
  executions, proposing a prior that faster plans will tend to stay on the
  faster side and slower plans will tend to stay on the slower side. This prior
  will not always hold and will not change the correctness of query execution
  and training data collection. The prior is used to increase the chances that
  the optimal timeout will be set to a lower value sooner and to keep
  consistenly poor plans at the end of the execution ordering, meaning they run
  with lower timeouts set by earlier executions.
  """

  def __init__(self, plan_count: int):
    self._plan_count = plan_count
    self._executions = dict.fromkeys(range(plan_count), 0)
    self._plan_index_invariant = set()
    self._current_default_plan_index = None
    self._plan_index_to_has_null = collections.defaultdict(lambda: False)

  def _reset_invariant_checker(self):
    self._plan_index_invariant = set(range(self._plan_count))

  def add_execution(self, plan_index: int,
                    execution_latency_ms: Optional[float]) -> None:
    """Adds execution latency for the provided plan_index.

    An execution must be added exactly once per plan_index before calling
    get_execution_order_and_reset().

    A plan is assumed to always return get_execution_latency values, and can
    switch to returning null values exactly once (in the beginning if it's
    skipped or EXPLAIN, or after greedy plan cover pruning).

    Args:
      plan_index: The plan index corresponding to the execution_latency_ms.
      execution_latency_ms: The latency to record for the plan_index.
    """
    assert plan_index in self._plan_index_invariant, (
        "Adding execution for an unknown plan index or a plan index for which "
        f"an execution already been added this iteration: {plan_index}")
    self._plan_index_invariant.remove(plan_index)

    if execution_latency_ms is not None:
      if plan_index != self._current_default_plan_index:
        assert not self._plan_index_to_has_null[plan_index], (
            f"Plan index {plan_index} had non-null result after null result.")
      self._executions[plan_index] += execution_latency_ms
    else:
      self._plan_index_to_has_null[plan_index] = True

  def get_execution_order_and_reset(self, default_plan_index: int) -> List[int]:
    """Returns the execution order based on past execution latencies.

    The default plan index will always be placed first in the execution order,
    regardless of prior execution latency.

    This function may only be called after add_execution() has been called
    exactly once for every plan index. The structures enforcing this invariant
    are also reset for the next iteration of add_execution()s when this function
    is called.

    Args:
      default_plan_index: The index of the default plan, which will always come
        first in the order.

    Returns:
      A list containing the recommended order of plan executions for the current
      parameter based on the execution latencies of the past parameters.
    """
    assert not self._plan_index_invariant, (
        "Some plans do not yet have execution data for this iteration: "
        f"{self._plan_index_invariant}")
    self._reset_invariant_checker()
    self._current_default_plan_index = default_plan_index

    execution_order = [
        key for (key,
                 value) in sorted(self._executions.items(), key=lambda x: x[1])
    ]

    # Move default to the front so that it's handled first.
    execution_order.remove(default_plan_index)
    execution_order.insert(0, default_plan_index)
    return execution_order


def _get_execution_latency(results: JSON, results_key: str,
                           params_as_string: str,
                           plan_index: int) -> Tuple[Optional[float], bool]:
  """Returns the plan execution latency amended for timeouts.

  The execution latency is determined as follows:
    1) If the data is not latency data or skipped, return None.
    2) If there are no timeouts, return the median execution latency.
    3) If any timeouts exist, return the execution latency of the first
       execution (or its timeout if it timed_out as well). The first execution
       uses the default timeout and hence gives the best representation of how
       bad the execution latency may be. If we hit 3), we are already only
       trying to distinguish amongst poor plan options. We want to ensure that
       the worst options remain sorted last when using the
       PlanExecutionOrderManager.

  Args:
    results: JSON mapping parameters to query execution results.
    results_key: A string corresponding to the key that maps to the result data.
    params_as_string: The params used for the execution.
    plan_index: The index identifying the query plan.

  Returns:
    Tuple:
      1) The execution latency or None if the execution is one that does not
        produce latency data.
      2) A boolean of whether there were any timeouts across the execution
        iterations.
  """
  # Check if we're in an instance of 1).
  #
  # If the first value is a float, presume we have an execute_query_fn that
  # produces numerical (currently only latency) results. Alternative
  # execute_query_fns may, for example, produce EXPLAIN plans and computing
  # execution latency is not relevant for those cases for this usage.
  execution_results = results[params_as_string]["results"][plan_index]
  if results_key in execution_results[0]:
    first_value = execution_results[0][results_key]
    if not isinstance(first_value, (int, float)):
      return None, False

  if "skipped" in execution_results[0]:
    return None, False

  first_key = "timed_out" if "timed_out" in execution_results[0] else results_key
  first_execution_latency_ms = execution_results[0][first_key]
  execution_values = [
      result[results_key]
      for result in execution_results
      if results_key in result
  ]

  if _has_timeout(execution_results):
    # This is an instance of 3).
    return first_execution_latency_ms, True

  # This is an instance of 2).
  assert len(execution_values) == len(execution_results)
  median_duration_ms = np.median(execution_values)
  return median_duration_ms, False


def _validate_inputs(num_initial_default_executions: Optional[int] = None,
                     slowest_default_top_k: Optional[int] = None,
                     slowest_default_sample_size: Optional[int] = None) -> None:
  """Validate inputs for execute_training_data_queries.

  Args:
    num_initial_default_executions: How many parameters to initially execute
      default plans for to determine the tail latency parameters.
    slowest_default_top_k: Specifies how many of the slowest parameters to
      sample from.
    slowest_default_sample_size: How many of the slowest k parameters to sample.

  Raises:
    ValueError: If there there are any inconsistencies in the parameter
      reordering hyperparameters.
  """
  if num_initial_default_executions is not None:
    if slowest_default_top_k is None or slowest_default_sample_size is None:
      raise ValueError(
          "slowest_default_top_k and slowest_default_sample_size must be specified if num_initial_default_executions is specified."
      )
    else:
      if slowest_default_top_k > num_initial_default_executions:
        raise ValueError(
            "slowest_default_top_k cannot be greater than num_initial_default_executions."
        )
      if slowest_default_sample_size > slowest_default_top_k:
        raise ValueError(
            "slowest_default_sample_size cannot be greater than slowest_default_top_k."
        )


def execute_training_data_queries(
    query_id: str,
    templates: Any,
    parameter_values: Any,
    plan_hints: Any,
    iterations: int,
    batch_size: int,
    limit: Union[int, None],
    skip_indices: List[int],
    query_timeout_multiplier: float,
    query_timeout_min_ms: int,
    query_timeout_max_ms: int,
    execute_query_fn: Callable[[str, List[Any], Optional[int]],
                               Tuple[Any, Optional[int]]],
    checkpoint_results_fn: Callable[[str, Any, bool], None],
    results_key: str,
    num_initial_default_executions: Optional[int] = None,
    slowest_default_top_k: Optional[int] = None,
    slowest_default_sample_size: Optional[int] = None,
    plan_cover_num_params: Optional[int] = None,
    near_optimal_threshold: Optional[float] = None,
    num_params_threshold: Optional[float] = None,
    query_timeout_minimum_speedup_multiplier: float = 1.,
    seed: int = 0) -> None:
  """Executes SQL queries generated from parameters and all query plans.

  For each set of parameter values and each query plan hint, we execute a query
  using execute_query_fn. The query execution latencies and metadata are stored
  in dicts and then passed to the checkpoint_results_fn function.

  This function also implements tail latency parameter reordering, which moves
  a sample of parameters with high default execution latencies to the front.

  Args:
    query_id: The query id for which to execute queries.
    templates: A mapping from query id to the templatized SQL query text.
    parameter_values: A mapping from query id to all the parameter values to
      execute with for that query id.
    plan_hints: A mapping from query id to pg_hint_plan hints representing the
      set of query plans for execution.
    iterations: The number of times to execute the requested query.
    batch_size: The number of parameter_values to fully evaluate before calling
      checkpoint_results_fn.
    limit: The number of parameter_values to gather execution data for.
    skip_indices: List of plan indices to skip.
    query_timeout_multiplier: This factor is multiplied by the slowest execution
      time of the default query plan to provide an upper bound on the query
      execution time considered 'way too slow' during execution data collection
      for each set of parameter values. This input has an inverse multiplicative
      relationship with query_timeout_minimum_speedup_multiplier. The product
      will be clipped to [query_timeout_min_ms, query_timeout_max_ms].
    query_timeout_min_ms: The minimum timeout for each query execution to enable
      setting a low multiplier while balancing the risk of timeouts caused by
      system noise for very fast queries.
    query_timeout_max_ms: The maximum timeout for each query execution to
      provide a hard-cap on the cost of very slow query plans.
    execute_query_fn: The function to call to execute a query. The function is
      expected to accept the query to execute, the list of parameters, and
      (optionally) the timeout_ms. It is expected to return a tuple of any data
      as relevant to the execute_query_fn use-case and the rows produced count.
    checkpoint_results_fn: A callback function called at a period defined by
      batch_size as well as upon completion of all executions. The function is
      expected to accept the query_id, a JSON object of results, and bool flag
      specifying whether the results are data or metadata.
    results_key: A string corresponding to the key that maps to the result data.
    num_initial_default_executions: How many parameters to initially execute
      default plans for to determine the tail latency parameters.
    slowest_default_top_k: Specifies how many of the slowest parameters to
      sample from.
    slowest_default_sample_size: How many of the slowest k parameters to sample.
    plan_cover_num_params: If set, use this many of the first parameters to
      compute a plan cover, and only execute these plans for the remaining
      parameters.
    near_optimal_threshold: Defines what constitutes a near-optimal plan: if
      latency < this value * optimal latency.
    num_params_threshold: Requires that this proportion of parameters be
       covered by the plan cover.
    query_timeout_minimum_speedup_multiplier: This factor describes the minimum
      speed up expected from a candidate plan to be considered an alternative to
      the default. Plans that do not provide this speed up are considered timed
      out.
    seed: The random seed to use.
  """
  _validate_inputs(num_initial_default_executions, slowest_default_top_k,
                   slowest_default_sample_size)

  np.random.seed(seed)
  results = {}
  metadata = {}

  def get_hinted_query(plan_index) -> str:
    current_plan_hints = plan_hints[query_id][plan_index]
    hinted_query = utils.get_hinted_query(
        query=templates[query_id]["query"], hints=current_plan_hints["hints"])
    return hinted_query

  def compute_timeout(execution_latency_ms: Optional[float],
                      has_timeout: bool) -> Optional[int]:
    # If any iteration times out, then we presume this is most
    # likely suboptimal plan and ignore it when computing a candidate for a
    # tighter timeout. We also ignore a plan that is skipped.
    if execution_latency_ms is None or has_timeout:
      return None

    return int(
        np.clip(
            int(execution_latency_ms * query_timeout_multiplier),
            query_timeout_min_ms, query_timeout_max_ms))

  def execute_query_iterations(results: JSON,
                               hinted_query: str,
                               plan_index: int,
                               params: List[Any],
                               params_as_string: str,
                               param_index: int,
                               should_skip: bool = False,
                               default_timeout_ms: Optional[int] = None,
                               optimal_timeout_ms: Optional[int] = None):
    execution_results = []
    if should_skip:
      execution_results.append({"skipped": True})
    else:
      for i in range(iterations):
        # The first iteration, we use the default_timeout_ms since the first
        # execution of a query sometimes has spiky behavior. After that, we
        # use the optimal_timeout_ms. If plans aren't similar enough to the
        # optimal plan for this set of parameters, we do not need to spend
        # time completing the full execution.
        timeout_ms = default_timeout_ms if i == 0 else optimal_timeout_ms

        result, rows = execute_query_fn(hinted_query, params, timeout_ms)
        if result:
          execution_results.append({results_key: result})
          results[params_as_string]["rows"].append(rows)
        else:
          execution_results.append({"timed_out": timeout_ms})

    results[params_as_string]["results"][plan_index] = execution_results

    # Log the final execution to give a pulse of how things are going.
    logging.info("%d/%d: %s,%d,%s %s ms", param_index + 1, len(param_sets),
                 query_id, plan_index, params_as_string,
                 str(execution_results[-1]))

  plan_count = len(plan_hints[query_id])

  logging.info("Processing: %s with %d plan_hints, skipping %s.", query_id,
               plan_count, skip_indices)

  param_sets = parameter_values[query_id]

  def execute_default_plans_only():
    default_results = {}
    default_latencies = []
    for param_index, params in enumerate(param_sets):
      if param_index >= num_initial_default_executions:
        break

      default_plan_index = int(params["plan_index"])
      params = params["params"]
      params_as_string = _NAME_DELIMITER.join([str(p) for p in params])
      default_results[params_as_string] = collections.defaultdict(dict)
      default_results[params_as_string]["default"] = default_plan_index
      default_results[params_as_string]["results"] = [None] * plan_count
      default_results[params_as_string]["rows"] = []

      # Execute without any timeouts.
      execute_query_iterations(
          default_results,
          get_hinted_query(default_plan_index),
          default_plan_index,
          params=params,
          params_as_string=params_as_string,
          param_index=param_index)

      min_default_latency = np.min([
          d[results_key] for d in default_results[params_as_string]["results"]
          [default_plan_index]
      ])
      default_latencies.append((param_index, min_default_latency))
    return default_results, default_latencies

  # Maybe execute default plans to move slowest plans to the front.
  # Use separate results to simplify JSON checkpointing.
  default_results = {}
  default_latencies = []
  reordered_param_sets = param_sets
  if num_initial_default_executions is not None:
    default_results, default_latencies = execute_default_plans_only()

    # Reorder param sets and sample from top-k slowest default plans.
    slowest_params = sorted(default_latencies, key=lambda x: x[1], reverse=True)
    sample_range = min(len(param_sets), slowest_default_top_k)
    sample_size = min(len(param_sets), slowest_default_sample_size)
    sampled_slowest = np.random.choice(
        sample_range, size=sample_size, replace=False)
    sampled_indices = [slowest_params[idx][0] for idx in sampled_slowest]

    # Move the selected slowest params to the front.
    reorder = sampled_indices + [
        i for i in range(len(param_sets)) if i not in sampled_indices
    ]
    reordered_param_sets = [param_sets[i] for i in reorder]

  plan_cover = None
  execution_order_manager = PlanExecutionOrderManager(plan_count=plan_count)
  for param_index, params in enumerate(reordered_param_sets):
    if limit and param_index >= limit:
      break

    default_plan_index = int(params["plan_index"])
    params = params["params"]
    params_as_string = _NAME_DELIMITER.join([str(p) for p in params])

    logging.info("Parameter: %s", params_as_string)

    results[params_as_string] = collections.defaultdict(dict)
    results[params_as_string]["default"] = default_plan_index
    # Pre-populate the list of results per plan because the results may be
    # populated out of order. Specifically, the default plan is always populated
    # first.
    results[params_as_string]["results"] = [None] * plan_count

    execution_order = execution_order_manager.get_execution_order_and_reset(
        default_plan_index)
    results[params_as_string]["execution_order"] = execution_order

    # Compute plan set cover if necessary. After executing
    # plan_cover_num_params, we use the execution data collected so far to
    # compute a minimal pruned plan set, such that each parameter is likely to
    # still have a near-optimal plan. From this iteration onwards, only these
    # plans and the default plans will be executed.
    if param_index == plan_cover_num_params:
      plan_cover = get_greedy_plan_cover(
          results,
          results_key,
          near_optimal_threshold=near_optimal_threshold,
          num_params_threshold=num_params_threshold,
          num_params_limit=plan_cover_num_params)
      metadata[PLAN_COVER] = [
          int(entry[0]) for entry in sorted(
              plan_cover.items(), key=lambda x: x[1], reverse=True)
      ]

    optimal_timeout_ms = None
    default_timeout_ms = None

    # A holder to collect the rows. It will be verified and collapsed after all
    # the executions.
    results[params_as_string]["rows"] = []
    for plan_index in execution_order:
      # Reuse default plan results if we already executed it above.
      if (plan_index == default_plan_index and
          params_as_string in default_results):
        results[params_as_string]["results"][plan_index] = default_results[
            params_as_string]["results"][plan_index]
        results[params_as_string]["rows"] = default_results[params_as_string][
            "rows"]
      else:
        should_skip = plan_index in skip_indices
        if plan_cover:
          should_skip |= plan_index not in plan_cover
        should_skip &= (plan_index != default_plan_index)

        execute_query_iterations(
            results,
            get_hinted_query(plan_index),
            plan_index,
            params=params,
            params_as_string=params_as_string,
            param_index=param_index,
            should_skip=should_skip,
            default_timeout_ms=default_timeout_ms,
            optimal_timeout_ms=optimal_timeout_ms)

      execution_latency_ms, has_timeout = _get_execution_latency(
          results=results,
          results_key=results_key,
          params_as_string=params_as_string,
          plan_index=plan_index)

      execution_order_manager.add_execution(
          plan_index=plan_index, execution_latency_ms=execution_latency_ms)
      candidate_timeout_ms = compute_timeout(execution_latency_ms, has_timeout)

      if plan_index == default_plan_index:
        assert optimal_timeout_ms is None
        assert default_timeout_ms is None
        default_timeout_ms = candidate_timeout_ms
        optimal_timeout_ms = default_timeout_ms
        if default_timeout_ms:
          optimal_timeout_ms = int(
              np.clip(
                  int(default_timeout_ms /
                      query_timeout_minimum_speedup_multiplier),
                  query_timeout_min_ms, query_timeout_max_ms))
        results[params_as_string]["timeout_ms"] = default_timeout_ms
      elif candidate_timeout_ms:
        optimal_timeout_ms = min(optimal_timeout_ms, candidate_timeout_ms)

    assert None not in results[params_as_string]["results"]

    # Verify all row entries are the same and collapse rows entries to that
    # value.
    rows_set = set(results[params_as_string]["rows"])
    assert len(rows_set) == 1, (
        f"All executions for parameter {params_as_string} should return the same "
        f"number of rows: {results[params_as_string]}")
    results[params_as_string]["rows"] = next(iter(rows_set))

    if len(results) % batch_size == 0:
      checkpoint_results_fn(query_id, {query_id: results}, False)
      checkpoint_results_fn(query_id, {query_id: metadata}, True)

  checkpoint_results_fn(query_id, {query_id: results}, False)
  checkpoint_results_fn(query_id, {query_id: metadata}, True)

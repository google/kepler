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

"""Evaluation framework for Kepler models and baselines."""

import copy

from typing import Any, Dict, List, Optional, Set

from absl import logging
import numpy as np
import scipy.stats

from kepler.data_management import database_simulator
from kepler.data_management import workload
from kepler.model_trainer import trainer_util

# Publicly visible keys for evaluation results.
IMPROVEMENTS_RELATIVE = "improvements_relative"
REGRESSIONS_RELATIVE = "regressions_relative"
IMPROVEMENTS_ABSOLUTE = "improvements_absolute"
REGRESSIONS_ABSOLUTE = "regressions_absolute"
NUM_EQUIVALENT = "num_equivalent"
NUM_NEAR_OPTIMAL = "num_near_optimal"

MEAN_CANDIDATE_LATENCY = "mean_candidate_latency"
MEAN_DEFAULT_LATENCY = "mean_default_latency"
MEAN_OPTIMAL_LATENCY = "mean_optimal_latency"

CANDIDATE_SUBOPTIMALITY = "candidate_suboptimality"
DEFAULT_SUBOPTIMALITY = "default_suboptimality"
CANDIDATE_SPEEDUP = "candidate_speedup"

CANDIDATE_TAIL_SUBOPTIMALITY = "candidate_tail_suboptimality"
DEFAULT_TAIL_SUBOPTIMALITY = "default_tail_suboptimality"
CANDIDATE_TAIL_SPEEDUP = "candidate_tail_speedup"

CANDIDATE_TAIL_CDF_SPEEDUP = "candidate_tail_cdf_speedup"
OPTIMAL_TAIL_CDF_SPEEDUP = "optimal_tail_cdf_speedup"


def _format_percentile_key(percentile: float) -> str:
  return f"p{percentile}"


def get_candidate_latencies(database: database_simulator.DatabaseSimulator,
                            query_workload: workload.Workload,
                            plan_selections: np.ndarray) -> List[float]:
  """Computes workload query execution latencies based on the plan selections.

  Args:
    database: The database used to execute the queries.
    query_workload: The workload of parameter bindings to execute for a query.
    plan_selections: The selected plan corresponding to each parameter binding
      in the query_workload. A plan selection may be set to None to indicate
      using the default plan for a given parameter. The plan selections are the
      output of a candidate method for choosing plans per parameter, such as the
      predictions from a trained model or a baseline algorithm.

  Returns:
    A list containing an execution latency value for each parameter binding in
    the query_workload provided. The latencies are provided in the same order as
    the query_workload and plan selections.

  Raises:
    ValueError: If the query_workload query_log and plan_selections are not the
      same length.
  """
  if len(query_workload.query_log) != len(plan_selections):
    raise ValueError(
        "The query_workload query log must contain the same number of parameter "
        "bindings as the provided plan selections. Received "
        f"{len(query_workload.query_log)} parameter bindings and "
        f"{len(plan_selections)} plan selections.")

  candidate_latencies = []
  for query, selected_plan_id in zip(query_workload.query_log, plan_selections):
    candidate_latencies.append(
        database.execute_timed(
            database_simulator.PlannedQuery(
                query_id=query_workload.query_id,
                plan_id=selected_plan_id
                if selected_plan_id is not None else None,
                parameters=query.parameters))[0])
  return candidate_latencies


def get_default_latencies(database: database_simulator.DatabaseSimulator,
                          query_workload: workload.Workload) -> List[float]:
  """Computes workload query execution latencies based on default plans.

  Args:
    database: The database used to execute the queries.
    query_workload: The workload of parameter bindings to execute for a query.

  Returns:
    A list containing an execution latency value for each parameter binding in
    the workload provided. The latencies are provided in the same order as the
    workload.
  """
  return get_candidate_latencies(
      database=database,
      query_workload=query_workload,
      plan_selections=np.array([None] * len(query_workload.query_log)))


def get_optimal_latencies(
    client: database_simulator.DatabaseClient,
    query_workload: workload.Workload,
    kepler_plan_discoverer: workload.KeplerPlanDiscoverer) -> List[float]:
  """Computes workload query execution latencies based on optimal plans.

  Args:
    client: The database client used to execute the queries.
    query_workload: The workload of parameter bindings to execute for a query.
    kepler_plan_discoverer: The plan discovery mechanism that scopes the set of
      plans from which the optimal should be identified per parameter binding in
      query_workload.

  Returns:
    A list containing an execution latency value for each parameter binding in
    the workload provided. The latencies are provided in the same order as the
    workload.
  """

  if not query_workload.query_log:
    return []

  queries = workload.create_query_batch(kepler_plan_discoverer.plan_ids,
                                        query_workload)
  query_execution_df = client.execute_timed_batch(planned_queries=queries)
  parameter_count = len(query_workload.query_log[0].parameters)
  return list(
      query_execution_df.groupby(
          trainer_util.get_parameter_column_names(parameter_count),
          sort=False).min()["latency_ms"])


class CostBasedCandidatePlanSetEvaluator:
  """A tool to construct and evaluate candidate plan sets with optimizer costs.

  Attributes:
    num_plans: The max plan_set_size that can be requested when calling
      populate_cache_greedy.
  """

  def __init__(self, client: database_simulator.DatabaseClient,
               query_workload: workload.Workload,
               kepler_plan_discoverer: workload.KeplerPlanDiscoverer,
               use_default_plans_only: bool):
    """Set up the evaluator with full data access.

    Args:
      client: The database client used to execute the queries. The client must
        be initialized with both execution and explain data.
      query_workload: The workload of parameter bindings to use for the
        evaluator. All parameters bindings must have execution and explain data.
      kepler_plan_discoverer: The plan discovery mechanism scoped to all
        possible plans in the data set.
      use_default_plans_only: If true, only populate the candidate set of plans
        with default plans used across the workload.

    Raises:
      ValueError: If the query_workload.query_log is empty.
    """
    if not query_workload.query_log:
      raise ValueError(
          "Provide a non-empty query_workload to evaluate.")

    self._client = client
    self._query_workload = query_workload
    batch = workload.create_query_batch(kepler_plan_discoverer.plan_ids,
                                        self._query_workload)
    self._query_execution_df = self._client.execute_timed_batch(
        planned_queries=batch, get_total_cost=True)

    parameter_count = len(query_workload.query_log[0].parameters)
    self._parameter_column_names = trainer_util.get_parameter_column_names(
        parameter_count)
    if use_default_plans_only:
      self._candidate_plan_ids = set(self._query_execution_df[
          self._query_execution_df["is_default"]]["plan_id"])
    else:
      self._candidate_plan_ids = kepler_plan_discoverer.plan_ids

    self.num_plans = len(self._candidate_plan_ids)

  def populate_cache_greedy(self, plan_set_size: int) -> Set[int]:
    """Determines the best set of plans within the plan_set_size limit.

    An implementation of populateCache from the 2022 Vaidya et al paper:
    https://www.vldb.org/pvldb/vol15/p401-vaidya.pdf. For the most direct
    replication of the algorithm, use_default_plans_only should be set to True
    in init.

    The populateCache method from the paper presents an approach to build a set
    of candidate plans based on optimizer costs.  This function implements the
    greedy algorithm from section 5.1 for populateCache.
    * m = self._candidate_plan_ids, the candidate plan set for the cache.
    * K = plan_set_size, the final number of plans to select.
    * metric = geometric mean suboptimality, per section 5.1.

    Args:
      plan_set_size: The number of plans to select from the m choices.

    Returns:
      The specific plans selected to populate the plan cache.

    Raises:
      ValueError: If plan_set_size is not in [0, self.num_plans].
    """
    if plan_set_size < 0 or plan_set_size > self.num_plans:
      raise ValueError("The plan_set_size must be between "
                       f"[0, {self.num_plans}],"
                       f"but {plan_set_size} was provided.")

    plan_set = set()
    candidate_plan_ids = copy.copy(self._candidate_plan_ids)
    for _ in range(plan_set_size):

      best_candidate = None
      best_suboptimality_score = None
      # Procedure:
      # 1. Add each candidate to the current plan_set.
      # 2. Compute the scoring metric.
      # 3. Maintain which candidate provided the lowest score for the metric.
      # 4. Add the lowest scoring candidate to the plan_set. Repeat.
      for candidate in candidate_plan_ids:
        plan_set.add(candidate)

        plan_set_min_cost = self._query_execution_df[
            self._query_execution_df["plan_id"].isin(plan_set)].groupby(
                self._parameter_column_names).min()["total_cost"]
        overall_min_cost = self._query_execution_df.groupby(
            self._parameter_column_names).min()["total_cost"]
        suboptimality_score = scipy.stats.gmean(plan_set_min_cost /
                                                overall_min_cost)

        if (best_suboptimality_score is None or
            suboptimality_score < best_suboptimality_score):
          best_suboptimality_score = suboptimality_score
          best_candidate = candidate

        plan_set.remove(candidate)

      assert best_candidate is not None
      plan_set.add(best_candidate)
      candidate_plan_ids.remove(best_candidate)

    return plan_set

  def get_latencies(self, plan_set: Optional[Set[int]] = None) -> List[float]:
    """Computes execution latencies based on optimal plans within plan_set.

    This function allows us to measure and compare the empirical quality of
    candidate plan sets constructed using the populateCache algorithm.

    Args:
      plan_set: The candidate set of plans for all plan selections. If plan_set
        is None, all candidate plans based on init arguments are used.

    Returns:
      A list containing an execution latency value for each parameter binding in
      the workload provided in the init. The latencies are provided in the same
      order as the workload.
    """
    candidate_plan_cover = (
        list(plan_set) if plan_set is not None else self._candidate_plan_ids)
    query_execution_metadata = {
        self._query_workload.query_id: {
            "plan_cover": candidate_plan_cover
        }
    }
    plan_discoverer = workload.KeplerPlanDiscoverer(
        query_execution_metadata=query_execution_metadata)
    return get_optimal_latencies(
        client=self._client,
        query_workload=self._query_workload,
        kepler_plan_discoverer=plan_discoverer)


def _compute_tail_ratios(numerator: List[float], denominator: List[float],
                         percentiles: List[float]) -> Dict[str, float]:
  ratios = {}
  for percentile in percentiles:
    ratios[_format_percentile_key(percentile)] = np.percentile(
        numerator, percentile) / np.percentile(denominator, percentile)
  return ratios


def _compute_tail_cdf_ratios(numerator: List[float], denominator: List[float],
                             percentiles: List[float]) -> Dict[str, float]:
  ratios = {}
  numerator = np.array(numerator)
  denominator = np.array(denominator)
  for percentile in percentiles:
    idxs = np.where(numerator >= np.percentile(numerator, percentile))[0]
    ratios[_format_percentile_key(percentile)] = (np.sum(numerator[idxs]) /
                                                  np.sum(denominator[idxs]))
  return ratios


def evaluate(candidate_latencies: List[float],
             default_latencies: List[float],
             optimal_latencies: List[float],
             near_optimal_threshold: float = 0.95,
             improvement_threshold: float = 1.1,
             percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
  """Computes and prints evaluation results for the candidate.

  Currently computes the following:
    * Default vs optimal suboptimality ratio
    * Candidate vs optimal suboptimality ratio
    * Candidate speedup over default
    * Number of improvements (relative, absolute)
    * Number of regressions (relative, absolute)
    * Number of equivalent plans
    * Number of near optimal plans

  Args:
    candidate_latencies: A list of latencies per parameter binding produced by
      the candidate being evaluated. The candidate_latencies may be represent a
      model or a baseline.
    default_latencies: A list of latencies for the default plan for each
      parameter binding. The order is assumed to match candidate_latencies.
    optimal_latencies: A list of latencies for the optimal plan for each
      parameter binding. The order is assumed to match candidate_latencies.
    near_optimal_threshold: If latency * this value < optimal latency, we say
      this plan is near-optimal.
    improvement_threshold: In order to filter out noise, a plan must be this
      factor faster or slower to be defined as better or worse.
    percentiles: The percentiles for which to calculate tail suboptimality and
      speedup ratios.

  Returns:
    Dict containing evaluation stats for use in visualizations or other
    post-processing.

  Raises:
    ValueError: If candidate_latencies, default_latencies, and optimal_latencies
      are not all the same length. Also raises if they are empty. This likely
      reflects a bug and returning an empty result is not helpful.
  """
  if (len(default_latencies) != len(candidate_latencies) or
      len(optimal_latencies) != len(candidate_latencies)):
    raise ValueError(
        "All of candidate_latencies, default_latencies, and optimal_latencies "
        f"must be the same length, but lengths of {len(candidate_latencies)}, "
        f"{len(default_latencies)}, {len(optimal_latencies)} provided "
        "respectively.")

  if not candidate_latencies:
    raise ValueError("No latency values to evaluate")

  improvements_relative = []
  regressions_relative = []
  improvements_absolute = []
  regressions_absolute = []
  num_equivalent = 0
  num_near_optimal = 0

  for candidate_latency, default_latency, optimal_latency in zip(
      candidate_latencies, default_latencies, optimal_latencies):
    if default_latency / candidate_latency > improvement_threshold:
      improvements_relative.append(default_latency / candidate_latency)
      improvements_absolute.append(default_latency - candidate_latency)
    elif candidate_latency / default_latency > improvement_threshold:
      regressions_relative.append(candidate_latency / default_latency)
      regressions_absolute.append(candidate_latency - default_latency)
    else:
      num_equivalent += 1
    if optimal_latency / candidate_latency > near_optimal_threshold:
      num_near_optimal += 1

  assert len(improvements_absolute) == len(improvements_relative)
  assert len(regressions_absolute) == len(regressions_relative)
  assert ((len(improvements_relative) + len(regressions_relative) +
           num_equivalent) == len(default_latencies))

  if percentiles is None:
    percentiles = [50, 90, 95, 99]

  candidate_tail_suboptimality = _compute_tail_ratios(
      numerator=candidate_latencies,
      denominator=optimal_latencies,
      percentiles=percentiles)

  default_tail_suboptimality = _compute_tail_ratios(
      numerator=default_latencies,
      denominator=optimal_latencies,
      percentiles=percentiles)

  candidate_tail_cdf_speedup = _compute_tail_cdf_ratios(
      numerator=default_latencies,
      denominator=candidate_latencies,
      percentiles=percentiles)

  optimal_tail_cdf_speedup = _compute_tail_cdf_ratios(
      numerator=default_latencies,
      denominator=optimal_latencies,
      percentiles=percentiles)

  candidate_tail_speedup = _compute_tail_ratios(
      numerator=default_latencies,
      denominator=candidate_latencies,
      percentiles=percentiles)

  mean_candidate_latency = np.mean(candidate_latencies)
  mean_default_latency = np.mean(default_latencies)
  mean_optimal_latency = np.mean(optimal_latencies)

  candidate_suboptimality = mean_candidate_latency / mean_optimal_latency
  default_suboptimality = mean_default_latency / mean_optimal_latency
  candidate_speedup = mean_default_latency / mean_candidate_latency

  logging.info(
      "Mean default latency: %.2f, "
      "mean candidate latency: %.2f, "
      "mean optimal latency: %.2f", mean_default_latency,
      mean_candidate_latency, mean_optimal_latency)

  logging.info(
      "Candidate suboptimality ratio: %.3f, "
      "default suboptimality ratio: %.3f, "
      "candidate speedup: %.3f", candidate_suboptimality,
      default_suboptimality, candidate_speedup)
  logging.info(
      "Total evaluated points: %d, num better: %d, num worse: %d, "
      "num equivalent: %d, num near optimal: %d", len(candidate_latencies),
      len(improvements_relative), len(regressions_relative), num_equivalent,
      num_near_optimal)
  if improvements_relative:
    logging.info("Relative Improvements: mean: %.3f, max: %.3f",
                 np.mean(improvements_relative), np.max(improvements_relative))
    logging.info("Absolute Improvements: mean: %.3f, max: %.3f",
                 np.mean(improvements_absolute), np.max(improvements_absolute))
  else:
    logging.info("No improvements!")

  if regressions_relative:
    logging.info("Relative Regressions: mean: %.3f, max: %.3f",
                 np.mean(regressions_relative), np.max(regressions_relative))
    logging.info("Absolute Regressions: mean: %.3f, max: %.3f",
                 np.mean(regressions_absolute), np.max(regressions_absolute))
  else:
    logging.info("No regressions!")

  return {
      IMPROVEMENTS_RELATIVE: improvements_relative,
      REGRESSIONS_RELATIVE: regressions_relative,
      IMPROVEMENTS_ABSOLUTE: improvements_absolute,
      REGRESSIONS_ABSOLUTE: regressions_absolute,
      NUM_EQUIVALENT: num_equivalent,
      NUM_NEAR_OPTIMAL: num_near_optimal,
      MEAN_CANDIDATE_LATENCY: mean_candidate_latency,
      MEAN_DEFAULT_LATENCY: mean_default_latency,
      MEAN_OPTIMAL_LATENCY: mean_optimal_latency,
      CANDIDATE_SUBOPTIMALITY: candidate_suboptimality,
      DEFAULT_SUBOPTIMALITY: default_suboptimality,
      CANDIDATE_SPEEDUP: candidate_speedup,
      CANDIDATE_TAIL_SUBOPTIMALITY: candidate_tail_suboptimality,
      DEFAULT_TAIL_SUBOPTIMALITY: default_tail_suboptimality,
      CANDIDATE_TAIL_CDF_SPEEDUP: candidate_tail_cdf_speedup,
      OPTIMAL_TAIL_CDF_SPEEDUP: optimal_tail_cdf_speedup,
      CANDIDATE_TAIL_SPEEDUP: candidate_tail_speedup
  }

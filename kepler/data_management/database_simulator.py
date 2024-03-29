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

"""Simulates query execution to get query execution latency.

The DatabaseSimulator enables rapid iteration on and evaluation of machine
learning ideas and experiments. Provided execution data defines the universe of
all possible queries in the environment. When a query from this universe is
"executed", the latency of that execution is returned immediately using the
(pre-collected) provided execution data.

The DatabaseClient builds functionality above the DatabaseSimulator to batch the
execution of many query plans and parameters bindings to produce training data
for downstream ML tasks.

The DatabaseClient accepts a list of PlannedQueries, and so the user is not
required to execute the crossproduct of plan ids and parameter bindings. The
crossproduct is expected to be a common use-case, however, and so a helper
function is provided to generate the appropriate input for execute_timed_batch.


```
Typical usage:

plan_ids = workload.KeplerPlanDiscoverer(query_execution_data).plan_ids
workload_generator = workload.WorkloadGenerator(query_execution_data)
my_workload = workload_generator.random_sample(1000)

client = database_simulator.DatabaseClient(query_execution_data)

batch = workload.create_query_batch(plan_ids, my_workload)
results = client.execute_timed_batch(planned_queries=batch)
```
"""

import copy
import dataclasses
import enum

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = "####"

_DEFAULT = "default"
_EXPLAINS = "explains"
_TOTAL_COST = "total_cost"


class LatencyEstimator(str, enum.Enum):
  MIN = "min"
  MAX = "max"
  MEDIAN = "median"

ESTIMATOR_MAP = {
    LatencyEstimator.MIN: np.min,
    LatencyEstimator.MAX: np.max,
    LatencyEstimator.MEDIAN: np.median
}


@dataclasses.dataclass
class PlannedQuery:
  """A QueryInstance assigned a plan with which to be executed.

  Attributes:
    query_id: The identifier for the query template.
    plan_id: The identifier for which plan to execute. A value of None will
      execute the default plan.
    parameters: The parameter bindings with which to execute.
  """
  query_id: str
  plan_id: Optional[int]
  parameters: List[str]


def _fetch_plan_id(default_plan_id: int, plan_id: Optional[int],
                   max_possible_plan_id: int) -> int:
  """Resolves default plan queries to plan id (if necessary) and validates."""
  if plan_id is None:
    plan_id = default_plan_id

  if plan_id < 0 or plan_id > max_possible_plan_id:
    raise ValueError(
        f"Provided plan id ({plan_id}) does not refer to a recognized plan.")

  return plan_id


def _is_plan_skipped(stats: Any, plan_id: int) -> bool:
  plan_results = stats["results"][plan_id]
  return any(["skipped" in element for element in plan_results])


def _is_plan_cover_plan_skipped(stats: Any,
                                plan_cover: List[int]) -> Optional[int]:
  for plan_id in plan_cover:
    if _is_plan_skipped(stats, plan_id):
      return plan_id

  return None


class DatabaseSimulator:
  """Simulates database interactions backed by provided query execution data.

  The caller to execute_timed must only use parameter bindings found in the
  backing execution data. The execution data is expected to represent a single
  query template.

  Attributes:
    query_id: The identifier for the query template.
    execution_count: The int number of queries executed, ie the number of times
      execute_timed() was called.
    execution_cost_ms: The float total actual execution latency for the queries
      sent to execute_timed().
  """

  def __init__(self,
               query_execution_data: Any,
               query_execution_metadata: Any,
               estimator: LatencyEstimator,
               query_explain_data: Optional[Any] = None):
    """Restructures provided data for fast look-ups.

    Args:
      query_execution_data: Execution data structure that defines all the known
        information regarding query plans, parameter bindings, and latencies.
        The format is a series of nested dicts, typically parsed from a JSON
        file.
      query_execution_metadata: Metadata from query execution that is expected
        to contain a "plan_cover" entry.
      estimator: Specifies how to compute the estimated runtime from a list of
        latencies from repeated executions. Must be contained in ESTIMATOR_MAP.
      query_explain_data: Explain plan data structure that contains only the
        plan total cost produced by the database. The format is a series of
        nested dicts, typically parsed from a JSON file. If provided, it must at
        least contain an entry for every parameter and plan in
        query_execution_data.

    Raises:
      ValueError: If query_execution_data, query_execution_metadata, or
        query_explain_data do not have the expected structure or are not
        consistent with each other. Also raise if execution data violates an
        invariant regarding which plans having corresponding execution data.
    """
    if len(query_execution_data) != 1:
      raise ValueError("Unexpected data format.")
    if len(query_execution_metadata) != 1:
      raise ValueError("Unexpected metadata format.")
    if query_explain_data is not None and len(query_explain_data) != 1:
      raise ValueError("Unexpected explain data format.")
    self.query_id = next(iter(query_execution_data))
    if self.query_id not in query_execution_metadata:
      raise ValueError(
          f"Query id mismatch between data arguments. Found {self.query_id} in "
          f"query_execution_data and {next(iter(query_execution_metadata))} in "
          "query_execution_metadata)")
    if (query_explain_data is not None and
        self.query_id not in query_explain_data):
      raise ValueError(
          f"Query id mismatch between data arguments. Found {self.query_id} in "
          f"query_execution_data and {next(iter(query_explain_data))} in "
          "query_explain_data)")
    data_mapping = query_execution_data[self.query_id]
    plan_cover = query_execution_metadata[self.query_id]["plan_cover"]

    self._table = {}
    for parameters_as_key, stats in data_mapping.items():
      if _EXPLAINS in stats:
        raise ValueError(f"Execution data contains key \"{_EXPLAINS}\", which "
                         "is reserved for query_explain_data")

      if "default_timed_out" in stats["results"]:
        continue

      # We rsplit since there is a param value with c#; in general our delimiter
      # of #### won't work if there are both params that begin and end with #.
      params_as_tuple = tuple(parameters_as_key.rsplit(_NAME_DELIMITER))

      if _is_plan_skipped(stats=stats, plan_id=stats[_DEFAULT]):
        raise ValueError(f"Default plan {stats[_DEFAULT]} skipped for params:"
                         f"{params_as_tuple}")

      plan_cover_skip_check_id = _is_plan_cover_plan_skipped(
          stats=stats, plan_cover=plan_cover)
      if plan_cover_skip_check_id is not None:
        raise ValueError(
            f"Plan id {plan_cover_skip_check_id} from the plan cover "
            f"{plan_cover} skipped for params: {params_as_tuple}")

      self._table[params_as_tuple] = copy.deepcopy(stats)

      # Merge explain data, if provided.
      if query_explain_data:
        if parameters_as_key not in query_explain_data[self.query_id]:
          raise ValueError(
              "query_explain_data must contain all parameter keys found in "
              f"query_execution_data, but {parameters_as_key} not found.")
        explains_data = query_explain_data[self.query_id][parameters_as_key]
        self._table[params_as_tuple][_EXPLAINS] = copy.deepcopy(
            explains_data["results"])

    self._estimator = estimator

    self.execution_count = 0
    self.execution_cost_ms = 0

  def _fetch_entry(self, planned_query: PlannedQuery) -> Any:
    """Validates the planned_query and fetches the corresponding data entry."""

    if self.query_id != planned_query.query_id:
      raise ValueError(
          f"Database Simulator is for query template {self.query_id} but "
          f"planned_query has template {planned_query.query_id}")
    if len(planned_query.parameters) != len(next(iter(self._table))):
      raise ValueError("All parameter bindings must be provided.")

    table_key = tuple(planned_query.parameters)
    if table_key not in self._table:
      raise ValueError(
          f"Out-of-universe query with plan {planned_query.plan_id} and "
          f"parameters: {planned_query.parameters}")

    return self._table[table_key]

  def get_plan_id(self, planned_query: PlannedQuery) -> int:
    """Gets the plan id corresponding to this query.

    Args:
      planned_query: The parameters and query plan for which to get the plan id.

    Returns:
      The plan id corresponding to this query.

    Raises:
      ValueError: If the planned_query is malformed or cannot be answered by
        this DatabaseSimulator instance.
    """
    entry = self._fetch_entry(planned_query)
    return _fetch_plan_id(
        default_plan_id=entry[_DEFAULT],
        plan_id=planned_query.plan_id,
        max_possible_plan_id=len(entry["results"]) - 1)

  def execute_explain(self, planned_query: PlannedQuery) -> Tuple[float, int]:
    """Returns the total cost from EXPLAIN for the requested planned query.

    Args:
      planned_query: The parameters and query plan for which to get total cost.

    Returns:
      The optimizer total cost for the requested configuration in planned_query.

      The plan_id corresponding to the explain total cost result is also
      returned. This is relevant when the planned_query requests the default
      plan since downstream analysis requires the plan id corresponding to the
      default plan.

    Raises:
      ValueError: If called without providing query_explain_data in the init.
        Also raises if the planned_query is malformed or cannot be answered by
        this DatabaseSimulator instance.
    """
    entry = self._fetch_entry(planned_query)
    if _EXPLAINS not in entry:
      raise ValueError(
          "Called execute_explain without providing query_explain_data in the "
          "init().")

    plan_id = _fetch_plan_id(
        default_plan_id=entry[_DEFAULT],
        plan_id=planned_query.plan_id,
        max_possible_plan_id=len(entry["results"]) - 1)
    return entry[_EXPLAINS][plan_id][0][_TOTAL_COST], plan_id

  def execute_timed(self, planned_query: PlannedQuery) -> Tuple[float, bool]:
    """Simulates the execution of the requested planned query to fetch timing.

    Args:
      planned_query: The parameters and query plan for which to simulate
        execution.

    Returns:
      The execution latency retrieved for the requested configuration in
      planned_query. That execution latency is expected to reflect previously
      collected empirical data.

      The database client also identifies whether the requested query plan was
      the database's default choice of query plan for the provided parameters.

    Raises:
      ValueError: If called without providing query_explain_data in the init.
        Also raises if the planned_query is malformed or cannot be answered by
        this DatabaseSimulator instance.
    """
    entry = self._fetch_entry(planned_query)
    plan_id = _fetch_plan_id(
        default_plan_id=entry[_DEFAULT],
        plan_id=planned_query.plan_id,
        max_possible_plan_id=len(entry["results"]) - 1)

    if _is_plan_skipped(stats=entry, plan_id=plan_id):
      raise ValueError(
          "Cannot execute query. No execution data was provided for plan "
          f"{plan_id} with parameter binding: {planned_query.parameters}")

    plan_results = entry["results"][plan_id]
    is_default = entry[_DEFAULT] == plan_id

    # If there are timeouts, set the time to the highest observed runtime.
    # In some cases, this may be the first iteration, which has a higher timeout
    # threshold.
    if any(["timed_out" in d for d in plan_results]):
      latency = np.max([next(iter(d.values())) for d in plan_results])
    else:
      estimator_func = ESTIMATOR_MAP[self._estimator]
      latency = estimator_func([d["duration_ms"] for d in plan_results])

    self.execution_count += 1
    self.execution_cost_ms += latency
    return float(latency), is_default


class DatabaseClient:
  """Provides added functionality to access the DatabaseSimulator.

  The DatabaseSimulator provides a simple and direct query execution
  interface. The DatabaseClient builds querying utilities on top of the
  DatabaseSimulator, such as executing queries in batch.
  """

  def __init__(self, database: DatabaseSimulator):
    self._database = database

  def execute_timed_batch(self,
                          planned_queries: List[PlannedQuery],
                          get_total_cost: bool = False) -> pd.DataFrame:
    """Executes the batch of requested queries to retrieve query stats.

    Args:
      planned_queries: List of PlannedQuerys to execute in batch.
      get_total_cost: Whether to get the total cost for this query. If not,
        uses a placeholder of -1 for the cost.

    Returns:
      A DataFrame containing a row of execution data and metadata for each
      PlannedQuery. The DataFrame contains a column for each parameter, named
      "param#" -- i.e., param0, param1, etc. Additionally, the DataFrame
      contains: plan_id, total_cost, latency_ms, and is_default.

    Raises:
      ValueError: If the batch is empty or any queries are malformed.
    """

    if not planned_queries:
      raise ValueError("Cannot execute empty batch.")

    data = []

    for planned_query in planned_queries:
      latency, is_default = self._database.execute_timed(planned_query)
      if get_total_cost:
        total_cost, plan_id = self._database.execute_explain(planned_query)
      else:
        total_cost = -1
        plan_id = self._database.get_plan_id(planned_query)
      data.append(planned_query.parameters +
                  [plan_id, total_cost, latency, is_default])

    first_query = next(iter(planned_queries))
    parameter_columns = [
        "param{}".format(i) for i in range(len(first_query.parameters))
    ]
    return pd.DataFrame(
        data=data,
        columns=parameter_columns +
        ["plan_id", _TOTAL_COST, "latency_ms", "is_default"])

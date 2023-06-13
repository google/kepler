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

"""Evaluates Kepler models integrated into a database system."""


from typing import Any, Callable, List, Tuple, Union


from kepler.data_management import workload

# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = "####"


def _get_params_as_string(params: List[Any]) -> str:
  return _NAME_DELIMITER.join([str(p) for p in params])


def evaluate_workload(
    workload_eval: workload.Workload,
    template: Any,
    iterations: int,
    batch_size: int,
    limit: Union[int, None],
    execute_query_fn: Callable[[str, List[Any]], Tuple[float, int]],
    checkpoint_results_fn: Callable[[str, Any], None],
) -> None:
  """Executes SQL queries generated defined by workload_eval.

  For each set of parameter values, we execute a query using
  execute_query_fn. The query execution latencies are stored in dicts and then
  passed to the checkpoint_results_fn function.

  Args:
    workload_eval: The workload defining which parameter bindings to evaluate.
    template: The templatized SQL query text.
    iterations: The number of times to execute the requested query.
    batch_size: The number of parameter_values to fully evaluate before calling
      checkpoint_results_fn.
    limit: The number of parameter_values to gather execution data for.
    execute_query_fn: The function to call to execute a query. The function is
      expected to accept the query to execute and the list of parameters.  It is
      expected to return a tuple of the query execution latency in ms and the
      rows produced count.
    checkpoint_results_fn: A callback function called at a period defined by
      batch_size as well as upon completion of all executions. The function is
      expected to accept the query_id and a JSON object of results.
  """
  query_id = workload_eval.query_id
  kepler_enabled_query_template = f"/*+ {query_id} */ {template['query']}"

  results = {}
  for query_instance in workload_eval.query_log[:limit]:
    params_as_string = _get_params_as_string(query_instance.parameters)

    execution_results = []
    rows_produced = None
    for _ in range(iterations):
      latency_ms, rows = execute_query_fn(
          kepler_enabled_query_template, query_instance.parameters
      )
      execution_results.append({"duration_ms": latency_ms})

      if rows_produced is None:
        rows_produced = rows
      else:
        assert rows == rows_produced, (
            f"All executions for parameter {params_as_string} should return the"
            f" same number of rows. Got {rows} and {rows_produced} instead."
        )

    results[params_as_string] = {
        "default": 0,
        "results": [execution_results],
        "rows": rows_produced,
    }

    if len(results) % batch_size == 0:
      checkpoint_results_fn(query_id, {query_id: results})

  checkpoint_results_fn(query_id, {query_id: results})

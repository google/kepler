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

"""Distributed training data query execution.

This script speeds up training data collection by executing queries in
parallel. It is recommended to execute this script after computing the plan
cover, which currently requires running the first N queries serially.
"""
import functools
import json
import multiprocessing
import os
from typing import Any, List, Optional, Tuple

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import main_utils
from kepler.training_data_collection_pipeline import pg_execute_training_data_queries
from kepler.training_data_collection_pipeline import query_text_utils
from kepler.training_data_collection_pipeline import query_utils

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")

_QUERY_TEMPLATES_FILE = flags.DEFINE_string(
    "query_templates_file",
    None,
    "Path to file in which query templates are stored.",
)
flags.mark_flag_as_required("query_templates_file")
_PARAMETER_VALUES_FILE = flags.DEFINE_string(
    "parameter_values_file", None, "Parameter values file."
)
flags.mark_flag_as_required("parameter_values_file")
_PLAN_HINTS_FILE = flags.DEFINE_string(
    "plan_hints_file", None, "Plan hints file."
)
flags.mark_flag_as_required("plan_hints_file")

_EXECUTION_METHOD = flags.DEFINE_string(
    "execution_method",
    "regular",
    (
        "Which execution method to use: regular to simply time the latency,"
        " explain, or explain_analyze."
    ),
)
_ITERATIONS = flags.DEFINE_integer(
    "iterations",
    3,
    (
        "The number of iterations to execute query (query plan, parameter"
        " binding) pairing."
    ),
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    10,
    (
        "Batch of parameters for which to collect training data before"
        " checkpointing."
    ),
)

_QUERY_TIMEOUT_MULTIPLIER = flags.DEFINE_integer(
    "query_timeout_multiplier",
    5,
    (
        "This factor is multiplied by the median execution time of the default"
        " query plan to provide an upper bound on the query execution time"
        " considered 'way too slow' during execution data collection for each"
        " set of parameter values. The product will be clipped to"
        " [query_timeout_min_ms, query_timeout_max_ms]."
    ),
)

_QUERY_TIMEOUT_MIN_MS = flags.DEFINE_integer(
    "query_timeout_min_ms",
    200,
    (
        "The minimum timeout for each query execution to enable setting a low"
        " multiplier while balancing the risk of timeouts caused by system"
        " noise for very fast queries."
    ),
)

_QUERY_TIMEOUT_MAX_MS = flags.DEFINE_integer(
    "query_timeout_max_ms",
    60 * 1000,
    (
        "The maximum timeout for each query execution to provide a hard-cap on"
        " the cost of very slow query plans."
    ),
)

_VERIFICATION_FILE = flags.DEFINE_string(
    "verification_file",
    None,
    (
        "File containing verification results. If specified, we will only"
        " execute hints for which there are no failures."
    ),
)

_NUM_PROCESSES = flags.DEFINE_integer(
    "num_processes", 1, "Number of processes to distribute execution over."
)

_QUERY = flags.DEFINE_string("query", None, "Specific query id to execute.")

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Directory to store execution results."
)
flags.mark_flag_as_required("output_dir")


def execute_query(
    query_manager: query_utils.QueryManager,
    query: str,
    params: List[Any],
    timeout_ms: Optional[int] = None,
) -> Tuple[Optional[float], Optional[int]]:
  return query_manager.execute_timed(query, params, timeout_ms)


def execute_query_local(
    query_manager: query_utils.QueryManager,
    query: str,
    params: List[Any],
    timeout_ms: Optional[int] = None,
) -> Tuple[Optional[float], Optional[int]]:
  return query_manager.execute_timed_local(query, params, timeout_ms)


def execute_explain(
    query_manager: query_utils.QueryManager,
    query: str,
    params: List[Any],
    _: Optional[int] = None,
) -> Tuple[Any, None]:
  return query_manager.get_query_plan(query, params), None


def execute_explain_analyze(
    query_manager: query_utils.QueryManager,
    query: str,
    params: List[Any],
    _: Optional[int] = None,
) -> Tuple[Any, None]:
  return query_manager.get_query_plan_and_execute(query, params), None


def _execute_func(args, **kwargs):
  batch_index, parameter_values = args
  return pg_execute_training_data_queries.execute_training_data_queries(
      batch_index, parameter_values, **kwargs
  )


def main(unused_argv):
  db_config = query_utils.DatabaseConfiguration(
      dbname=_DATABASE.value, user=_USER.value, password=_PASSWORD.value
  )
  query_manager = query_utils.QueryManager(db_config)
  query_utils.save_postgres_config_info(query_manager, _OUTPUT_DIR.value)

  with open(_PLAN_HINTS_FILE.value) as f:
    plan_hints = json.load(f)

  query_id_to_skip_indices = main_utils.get_skip_indices(
      plan_hints, _VERIFICATION_FILE.value
  )

  with open(_PARAMETER_VALUES_FILE.value) as f:
    parameter_values = json.load(f)

  with open(_QUERY_TEMPLATES_FILE.value) as f:
    templates = json.load(f)

  query_ids = [_QUERY.value] if _QUERY.value else plan_hints.keys()

  output_subdir = os.path.join(_OUTPUT_DIR.value, "execution_output")
  os.makedirs(output_subdir, exist_ok=True)

  def get_filename(query_id: str, is_metadata: bool) -> str:
    type_token = "_metadata" if is_metadata else ""
    return os.path.join(
        output_subdir, f"{_DATABASE.value}_{query_id}{type_token}.json"
    )

  def checkpoint_results(
      query_id: str, results: Any, is_metadata: bool
  ) -> None:
    with open(get_filename(query_id, is_metadata), "w") as f:
      json.dump(results, f)

  execution_method_map = {
      "regular": (execute_query_local, "duration_ms"),
      "explain": (execute_explain, "explain_output"),
      "explain_analyze": (execute_explain_analyze, "explain_analyze_output"),
  }
  execution_method, results_key = execution_method_map[_EXECUTION_METHOD.value]

  for query_id in query_ids:
    # Load previous results and metadata.
    previous_results = {query_id: {}}
    previous_metadata = {query_id: {}}
    results_file = get_filename(query_id, False)
    metadata_file = get_filename(query_id, True)
    exists_results = os.path.exists(results_file)
    exists_metadata = os.path.exists(metadata_file)
    assert exists_metadata

    if exists_results:
      with open(results_file) as f:
        previous_results = json.load(f)
    with open(metadata_file) as f:
      previous_metadata = json.load(f)

    # Compute indices to skip.
    skip_indices = set(query_id_to_skip_indices.get(query_id, []))
    plan_cover = previous_metadata[query_id].get("plan_cover", [])
    for i in range(len(plan_hints[query_id])):
      if i not in plan_cover:
        skip_indices.add(i)

    skip_indices = list(skip_indices)

    # Filter previously-executed query parameter values and split into batches.
    batch_size = _BATCH_SIZE.value
    parameter_value_batches = []
    query_parameter_values = parameter_values[query_id]

    def filter_nonexecuted(
        param_set, previous_params=previous_results[query_id]
    ):
      return (
          query_text_utils.get_params_as_string(param_set["params"])
          not in previous_params
      )

    query_parameter_values = list(
        filter(filter_nonexecuted, query_parameter_values)
    )
    batch_indexes = list(range(0, len(query_parameter_values), batch_size))
    for i in batch_indexes:
      parameter_value_batches.append(
          {query_id: query_parameter_values[i : i + batch_size]}
      )

    results = previous_results[query_id]
    with multiprocessing.Pool(
        processes=_NUM_PROCESSES.value,
        initializer=pg_execute_training_data_queries.init_per_process_global_query_manager,
        initargs=[db_config],
    ) as pool:
      kwargs = {
          "query_id": query_id,
          "templates": templates,
          "plan_hints": plan_hints,
          "iterations": _ITERATIONS.value,
          "batch_size": _BATCH_SIZE.value,
          "skip_indices": skip_indices,
          "query_timeout_multiplier": _QUERY_TIMEOUT_MULTIPLIER.value,
          "query_timeout_min_ms": _QUERY_TIMEOUT_MIN_MS.value,
          "query_timeout_max_ms": _QUERY_TIMEOUT_MAX_MS.value,
          "execute_query_fn": execution_method,
          "checkpoint_results_fn": None,
          "results_key": results_key,
          "total_num_params": len(query_parameter_values),
          "print_skips": False,
      }

      for _, (batch_results, _) in enumerate(
          pool.imap(
              func=functools.partial(_execute_func, **kwargs),
              iterable=zip(batch_indexes, parameter_value_batches),
              chunksize=1,
          )
      ):
        results.update(batch_results)
        checkpoint_results(query_id, {query_id: results}, False)


if __name__ == "__main__":
  app.run(main)

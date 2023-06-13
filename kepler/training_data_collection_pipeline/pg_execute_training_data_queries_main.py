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

"""Orchestrates training data collection for Kepler.

Results in JSON format are saved to the directory indicated by the output flag.
"""
import json
import os
from typing import Any, List, Optional, Tuple

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import main_utils
from kepler.training_data_collection_pipeline import pg_execute_training_data_queries
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
_LIMIT = flags.DEFINE_integer(
    "limit",
    None,
    "Limit the number of parameters per query to gather execution data for.",
)

_QUERY_TIMEOUT_MULTIPLIER = flags.DEFINE_integer(
    "query_timeout_multiplier",
    5,
    (
        "This factor is multiplied by the median execution time of the default"
        " query plan to provide an upper bound on the query execution time"
        " considered 'way too slow' during execution data collection for each"
        " set of parameter values. This input has an inverse multiplicative"
        " relationship with query_timeout_minimum_speedup_multiplier. The"
        " product will be clippedto [query_timeout_min_ms,"
        " query_timeout_max_ms]."
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

_QUERY_TIMEOUT_MINIMUM_SPEEDUP_MULTIPLIER = flags.DEFINE_integer(
    "query_timeout_minimum_speedup_multiplier",
    1,
    (
        "This factor describes the minimum speed up expected from a candidate"
        " plan to be considered an alternative to the default. Plans that do"
        " not provide this speed up are considered timed out. This input has an"
        " inverse multiplicative relationship with query_timeout_multiplier."
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

_NUM_INITIAL_DEFAULT_EXECUTIONS = flags.DEFINE_integer(
    "num_initial_default_executions",
    None,
    (
        "How many parameters to initially execute default plans for to"
        " determine the tail latency parameters."
    ),
)

_SLOWEST_DEFAULT_TOP_K = flags.DEFINE_integer(
    "slowest_default_top_k",
    None,
    "Specifies how many of the slowest parameters to sample from.",
)

_SLOWEST_DEFAULT_SAMPLE_SIZE = flags.DEFINE_integer(
    "slowest_default_sample_size",
    None,
    "How many of the slowest k parameters to sample.",
)

_PLAN_COVER_NUM_PARAMS = flags.DEFINE_integer(
    "plan_cover_num_params",
    None,
    (
        "Use the first N parameters to compute the plan cover. If not"
        " specified, plan cover pruning won't be used."
    ),
)

_NEAR_OPTIMAL_THRESHOLD = flags.DEFINE_float(
    "near_optimal_threshold",
    None,
    (
        "Defines what constitutes a near-optimal plan: if latency < "
        "this value * optimal latency."
    ),
)

_NUM_PARAMS_THRESHOLD = flags.DEFINE_float(
    "num_params_threshold",
    None,
    "Requires that this proportion of parameters be covered by the plan cover.",
)

_RESULTS_FILE = flags.DEFINE_string(
    "results_file",
    None,
    "File containing previous results. Used to resume execution.",
)

_METADATA_FILE = flags.DEFINE_string(
    "metadata_file",
    None,
    "File containing previous metadata. Used to resume execution.",
)

_QUERY = flags.DEFINE_string("query", None, "Specific query id to execute.")

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Directory to store execution results."
)
flags.mark_flag_as_required("output_dir")


def main(unused_argv):
  query_manager = query_utils.QueryManager(
      query_utils.DatabaseConfiguration(
          dbname=_DATABASE.value, user=_USER.value, password=_PASSWORD.value
      )
  )
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

  def checkpoint_results(query_id: str, results: Any,
                         is_metadata: bool) -> None:
    type_token = "_metadata" if is_metadata else ""
    with open(
        os.path.join(output_subdir,
                     f"{_DATABASE.value}_{query_id}{type_token}.json"),
        "w") as f:
      json.dump(results, f)

  def execute_query(
      unused_query_manager: query_utils.QueryManager,
      query: str,
      params: List[Any],
      timeout_ms: Optional[int] = None,
  ) -> Tuple[Optional[float], Optional[int]]:
    del unused_query_manager
    return query_manager.execute_timed(query, params, timeout_ms)

  def execute_explain(
      unused_query_manager: query_utils.QueryManager,
      query: str,
      params: List[Any],
      _: Optional[int] = None,
  ) -> Tuple[Any, None]:
    del unused_query_manager
    return query_manager.get_query_plan(query, params), None

  def execute_explain_analyze(
      unused_query_manager: query_utils.QueryManager,
      query: str,
      params: List[Any],
      _: Optional[int] = None,
  ) -> Tuple[Any, None]:
    del unused_query_manager
    return query_manager.get_query_plan_and_execute(query, params), None

  execution_method_map = {
      "regular": (execute_query, "duration_ms"),
      "explain": (execute_explain, "explain_output"),
      "explain_analyze": (execute_explain_analyze, "explain_analyze_output")
  }
  execution_method, results_key = execution_method_map[_EXECUTION_METHOD.value]

  previous_results = None
  previous_metadata = None
  if _RESULTS_FILE.value:
    with open(_RESULTS_FILE.value) as f:
      previous_results = json.load(f)

  if _METADATA_FILE.value:
    with open(_METADATA_FILE.value) as f:
      previous_metadata = json.load(f)

  for query_id in query_ids:
    pg_execute_training_data_queries.execute_training_data_queries(
        batch_index=0,
        parameter_values=parameter_values,
        query_id=query_id,
        templates=templates,
        plan_hints=plan_hints,
        iterations=_ITERATIONS.value,
        batch_size=_BATCH_SIZE.value,
        skip_indices=query_id_to_skip_indices.get(query_id, []),
        query_timeout_multiplier=_QUERY_TIMEOUT_MULTIPLIER.value,
        query_timeout_min_ms=_QUERY_TIMEOUT_MIN_MS.value,
        query_timeout_max_ms=_QUERY_TIMEOUT_MAX_MS.value,
        execute_query_fn=execution_method,
        checkpoint_results_fn=checkpoint_results,
        results_key=results_key,
        limit=_LIMIT.value,
        num_initial_default_executions=_NUM_INITIAL_DEFAULT_EXECUTIONS.value,
        slowest_default_top_k=_SLOWEST_DEFAULT_TOP_K.value,
        slowest_default_sample_size=_SLOWEST_DEFAULT_SAMPLE_SIZE.value,
        plan_cover_num_params=_PLAN_COVER_NUM_PARAMS.value,
        near_optimal_threshold=_NEAR_OPTIMAL_THRESHOLD.value,
        num_params_threshold=_NUM_PARAMS_THRESHOLD.value,
        query_timeout_minimum_speedup_multiplier=_QUERY_TIMEOUT_MINIMUM_SPEEDUP_MULTIPLIER.value,
        previous_results=previous_results,
        previous_metadata=previous_metadata,
    )

if __name__ == "__main__":
  app.run(main)

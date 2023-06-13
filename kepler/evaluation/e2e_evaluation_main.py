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

"""Orchestrates evaluating Kepler models integrated into a database system.

Results in JSON format are saved to the directory indicated by the output flag.
"""
import json
import os
from typing import Any, List, Optional, Tuple

from absl import app
from absl import flags

from kepler.data_management import workload
from kepler.evaluation import e2e_evaluation
from kepler.training_data_collection_pipeline import query_utils

_SET_MODEL_SERVER_PORT = "SET pg_hint_plan.kepler_port TO {port};"
_SET_MODEL_SERVER_HOST = "SET pg_hint_plan.kepler_host TO '127.0.0.1';"

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")

_MODEL_SERVER_PORT = flags.DEFINE_integer(
    "model_server_port", 30709, "The port for the model server to use."
)

_QUERY_TEMPLATES_FILE = flags.DEFINE_string(
    "query_templates_file",
    None,
    "Path to file in which query templates are stored.",
)
flags.mark_flag_as_required("query_templates_file")

_EXECUTION_DATA_FILE = flags.DEFINE_string(
    "execution_data_file", None, "Execution data file."
)
flags.mark_flag_as_required("execution_data_file")

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
    50,
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

_SEED = flags.DEFINE_integer(
    "seed", 0, "The seed to use when shuffling the workload before splitting."
)

_TRAIN_SPLIT = flags.DEFINE_float(
    "train_split",
    0.8,
    "The fraction of workload query instances to place in the training split.",
)

_QUERY = flags.DEFINE_string("query", None, "Specific query id to execute.")
flags.mark_flag_as_required("query")

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
  query_manager.execute_and_commit(
      _SET_MODEL_SERVER_PORT.format(port=_MODEL_SERVER_PORT.value)
  )
  query_manager.execute_and_commit(_SET_MODEL_SERVER_HOST)

  with open(_EXECUTION_DATA_FILE.value) as f:
    execution_data = json.load(f)
  workload_generator = workload.WorkloadGenerator(execution_data)
  full_workload = workload_generator.all()
  workload.shuffle(full_workload, _SEED.value)
  _, workload_eval = workload.split(
      full_workload, first_half_fraction=_TRAIN_SPLIT.value
  )

  with open(_QUERY_TEMPLATES_FILE.value) as f:
    templates = json.load(f)

  query_id = _QUERY.value

  output_subdir = os.path.join(_OUTPUT_DIR.value, "execution_output")
  os.makedirs(output_subdir, exist_ok=True)

  def checkpoint_results(query_id: str, results: Any) -> None:
    with open(
        os.path.join(
            output_subdir, f"{_DATABASE.value}_{query_id}_e2e_evaluation.json"
        ),
        "w",
    ) as f:
      json.dump(results, f)

  def execute_query(
      query: str, params: List[Any]
  ) -> Tuple[Optional[float], Optional[int]]:
    return query_manager.execute_timed(query, params)

  e2e_evaluation.evaluate_workload(
      workload_eval=workload_eval,
      template=templates[query_id],
      iterations=_ITERATIONS.value,
      batch_size=_BATCH_SIZE.value,
      limit=_LIMIT.value,
      execute_query_fn=execute_query,
      checkpoint_results_fn=checkpoint_results,
  )


if __name__ == "__main__":
  app.run(main)

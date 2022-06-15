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

The tool perturbs join cardinality estimates and collects the corresponding
EXPLAIN plans which include the total cost. This data is intended for
cardinality-estimate representation learning.

Results in JSON format are saved to the directory indicated by the output flag.
"""
import json
import os

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import pg_perturb_plan_cardinalities
from kepler.training_data_collection_pipeline import query_utils

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")
_HOST = flags.DEFINE_string("host", "localhost", "Database host.")
_SEED = flags.DEFINE_float("seed", 0, "Database random number seed.")

_QUERY_TEMPLATES_FILE = flags.DEFINE_string(
    "query_templates_file", None,
    "Path to file in which query templates are stored.")
flags.mark_flag_as_required("query_templates_file")
_PARAMETER_VALUES_FILE = flags.DEFINE_string("parameter_values_file", None,
                                             "Parameter values file.")
flags.mark_flag_as_required("parameter_values_file")
_PLAN_HINTS_FILE = flags.DEFINE_string("plan_hints_file", None,
                                       "Plan hints file.")
flags.mark_flag_as_required("plan_hints_file")

_CARDINALITY_MULTIPLIERS = flags.DEFINE_list(
    "cardinality_multipliers", None,
    "List of cardinality multipliers to apply when generating plans.")
flags.mark_flag_as_required("cardinality_multipliers")

_LIMIT = flags.DEFINE_integer(
    "limit", 1,
    "Limit the number of parameters per query to gather cost estimates for.")

_KEYS_TO_REMOVE = flags.DEFINE_list(
    "keys_to_remove", [],
    ("List of keys to filter from EXPLAIN plan JSON. Good candidates include "
     "\"Parallel Aware\", \"Relation Name\", \"Parent Relationship\""))

_QUERY = flags.DEFINE_string("query", None, "Specific query id to execute.")

_OUTPUT_DIR = flags.DEFINE_string("output_dir", None,
                                  "Directory to store execution results.")
flags.mark_flag_as_required("output_dir")


def main(unused_argv):
  query_manager = query_utils.QueryManager(
      query_utils.DatabaseConfiguration(
          dbname=_DATABASE.value, user=_USER.value, password=_PASSWORD.value))
  query_utils.save_postgres_config_info(query_manager, _OUTPUT_DIR.value)

  output_subdir = os.path.join(_OUTPUT_DIR.value, "execution_output")
  os.makedirs(output_subdir, exist_ok=True)

  with open(_PLAN_HINTS_FILE.value) as f:
    plan_hints = json.load(f)

  with open(_PARAMETER_VALUES_FILE.value) as f:
    parameter_values = json.load(f)

  with open(_QUERY_TEMPLATES_FILE.value) as f:
    templates = json.load(f)

  cardinality_multipliers = [
      float(multiplier) for multiplier in _CARDINALITY_MULTIPLIERS.value
  ]

  query_ids = [_QUERY.value] if _QUERY.value else plan_hints.keys()

  for query_id in query_ids:
    results = pg_perturb_plan_cardinalities.multiplicatively_perturb_plan_cardinalities(
        query_manager=query_manager,
        query_id=query_id,
        templates=templates,
        parameter_values=parameter_values,
        plan_hints=plan_hints,
        cardinality_multipliers=cardinality_multipliers,
        limit=_LIMIT.value,
        keys_to_remove=_KEYS_TO_REMOVE.value)

    with open(
        os.path.join(output_subdir, f"{_DATABASE.value}_{query_id}.json"),
        "w") as f:
      json.dump(results, f)


if __name__ == "__main__":
  app.run(main)

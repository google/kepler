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

"""Executes the collection of EXPLAIN plans for various analyses.

Results in JSON format are saved to the directory indicated by the output flag.
"""
import json
import os

from typing import Any

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import pg_execute_explain_tools
from kepler.training_data_collection_pipeline import query_utils


# Typing aliases.
JSON = Any

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
_EXTRACT_FUNCTION = flags.DEFINE_enum_class(
    "extract_function",
    pg_execute_explain_tools.ExplainExtractionFunction.TOTAL_COSTS.value,
    pg_execute_explain_tools.ExplainExtractionFunction,
    "Which EXPLAIN extraction function to use.")
_LIMIT = flags.DEFINE_integer(
    "limit", 1, "Limit the number of parameters per query to use for analysis.")
_CHUNKSIZE = flags.DEFINE_integer("chunksize", 100,
                                  "Multiprocessing chunksize.")

_QUERY = flags.DEFINE_string("query", None, "Specific query id to analyze.")

_OUTPUT_DIR = flags.DEFINE_string("output_dir", None,
                                  "Directory to store execution results.")
flags.mark_flag_as_required("output_dir")


def _save(data: JSON, output_subdir: str, query_id: str,
          filename_prefix: str) -> None:
  with open(
      os.path.join(output_subdir,
                   f"{filename_prefix}_{_DATABASE.value}_{query_id}.json"),
      "w") as f:
    json.dump(data, f)


def _calculate_plan_changing_cardinality_estimates(
    output_subdir: str,
    database_configuration: query_utils.DatabaseConfiguration, query_id: str,
    templates: JSON, parameter_values: JSON) -> None:
  """Calculates plan changing cardinalty estimates and writes out the result."""
  results = pg_execute_explain_tools.calculate_plan_changing_cardinality_estimates(
      database_configuration=database_configuration,
      query_id=query_id,
      templates=templates,
      parameter_values=parameter_values,
      limit=_LIMIT.value,
      multiprocessing_chunksize=_CHUNKSIZE.value)

  _save(
      data=results,
      output_subdir=output_subdir,
      query_id=query_id,
      filename_prefix="plan_changing_cardinality_estimates")


def _collect_explain_plan_info(
    output_subdir: str,
    database_configuration: query_utils.DatabaseConfiguration, query_id: str,
    templates: JSON, parameter_values: JSON, plan_hints: JSON,
    extract_function: pg_execute_explain_tools.ExplainExtractionFunction
) -> None:
  """Collects explain plan info and writes out the result."""
  results = pg_execute_explain_tools.collect_explain_plan_info(
      database_configuration=database_configuration,
      query_id=query_id,
      templates=templates,
      parameter_values=parameter_values,
      plan_hints=plan_hints,
      extract_function=extract_function,
      limit=_LIMIT.value,
      multiprocessing_chunksize=_CHUNKSIZE.value)

  _save(
      data=results,
      output_subdir=output_subdir,
      query_id=query_id,
      filename_prefix="explain_plan_total_costs")


def main(unused_argv):
  database_configuration = query_utils.DatabaseConfiguration(
      dbname=_DATABASE.value,
      user=_USER.value,
      password=_PASSWORD.value,
      host=_HOST.value,
      seed=_SEED.value)
  query_manager = query_utils.QueryManager(database_configuration)
  query_utils.save_postgres_config_info(query_manager, _OUTPUT_DIR.value)

  output_subdir = os.path.join(_OUTPUT_DIR.value, "explain_output")
  os.makedirs(output_subdir, exist_ok=True)

  with open(_PARAMETER_VALUES_FILE.value) as f:
    parameter_values = json.load(f)

  with open(_QUERY_TEMPLATES_FILE.value) as f:
    templates = json.load(f)

  plan_hints = None
  if _PLAN_HINTS_FILE.value:
    with open(_PLAN_HINTS_FILE.value) as f:
      plan_hints = json.load(f)

  query_ids = [_QUERY.value] if _QUERY.value else parameter_values.keys()

  for query_id in query_ids:
    if plan_hints:
      _collect_explain_plan_info(
          output_subdir=output_subdir,
          database_configuration=database_configuration,
          query_id=query_id,
          templates=templates,
          parameter_values=parameter_values,
          plan_hints=plan_hints,
          extract_function=_EXTRACT_FUNCTION.value)
    else:
      _calculate_plan_changing_cardinality_estimates(
          output_subdir=output_subdir,
          database_configuration=database_configuration,
          query_id=query_id,
          templates=templates,
          parameter_values=parameter_values)


if __name__ == "__main__":
  app.run(main)

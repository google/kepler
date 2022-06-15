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

"""Script for generating a set of query plans to use for a query template.

This step of the pipeline takes parameter values and query templates, and
produces a set of query plans as execution options for the query template. The
set of query plans are represented as pg_hint_plan hints.

The script supports several methods for generating the plan candidates, listed
in the _GENERATION_FUNCTION_MAP.
"""
import enum
import json
import os

from absl import app
from absl import flags
from absl import logging

from kepler.training_data_collection_pipeline import main_utils
from kepler.training_data_collection_pipeline import pg_generate_plan_candidates
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils


class GenerationFunction(enum.Enum):
  PG_CONFIGS = "pg_configs"
  ROW_NUM_EVOLUTION = "row_num_evolution"
  EXHAUSTIVE_CARDINALITY_PERTURBATIONS = "exhaustive_cardinality_perturbations"


_GENERATION_FUNCTION_MAP = {
    GenerationFunction.PG_CONFIGS:
        pg_generate_plan_candidates.get_query_plans,
    GenerationFunction.ROW_NUM_EVOLUTION:
        pg_generate_plan_candidates.generate_by_row_num_evolution,
    GenerationFunction.EXHAUSTIVE_CARDINALITY_PERTURBATIONS:
        pg_generate_plan_candidates
        .generate_by_exhaustive_cardinality_perturbations
}


def _supports_distributed_execution(
    generation_function: GenerationFunction) -> bool:
  return generation_function != GenerationFunction.EXHAUSTIVE_CARDINALITY_PERTURBATIONS

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")
_HOST = flags.DEFINE_string("host", "localhost", "Database host.")

_QUERY_PARAMS_FILE = flags.DEFINE_string(
    "query_params_file", None,
    "File containing parameterized queries with list of parameter values.")
flags.mark_flag_as_required("query_params_file")
_PARAMS_LIMIT = flags.DEFINE_integer(
    "params_limit", None,
    "The number of parameter values to use when generating plans.")
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory in which to store query plan hints and configs.")
flags.mark_flag_as_required("output_dir")

_PLANS_OUTPUT_FILE = flags.DEFINE_string(
    "plans_output_file", None,
    "File to store distinct plans per query. The file name is expected to end in .json"
)
flags.mark_flag_as_required("plans_output_file")
_PLAN_INDEX_SUFFIX = flags.DEFINE_string(
    "plan_index_suffix", "_plan_index.json",
    "Suffix of files to store plan indices in.")
_VERIFICATION_FAILURES_FILE = flags.DEFINE_string(
    "verification_failures_file", "verification_failures.json",
    "Filename of file to save verification failures.")
_CHUNKSIZE = flags.DEFINE_integer(
    "chunksize", 100, "How many params to include in each subprocess chunk.")
_KEYS_TO_REMOVE = flags.DEFINE_list(
    "keys_to_remove", [],
    ("List of keys to filter from EXPLAIN plan JSON. Good candidates include "
     "\"Parallel Aware\", \"Relation Name\", \"Parent Relationship\""))

_GENERATION_FUNCTION = flags.DEFINE_enum_class(
    "generation_function", GenerationFunction.PG_CONFIGS.value,
    GenerationFunction, "Which plan generation function to use.")
_SOFT_TOTAL_PLANS_LIMIT = flags.DEFINE_integer(
    "soft_total_plans_limit", None,
    "Soft limit on total number of plans to produce."
)
# Pg configs flags.
_CONFIG_STR = flags.DEFINE_string(
    "configs", "",
    "Comma-separated string of Postgres optimizer configuration parameters to toggle off."
)
# Row number evolution flags.
_MAX_PLANS_PER_PARAM = flags.DEFINE_integer(
    "max_plans_per_param", None,
    "Stop evolution after this number of plans is exceeded.")
_NUM_GENERATIONS = flags.DEFINE_integer(
    "num_generations", 3, "Number of generations of row number evolution.")
_NUM_MUTATIONS_PER_PLAN = flags.DEFINE_integer(
    "num_mutations_per_plan", 25, "Number of random mutations for each plan.")
_EXPONENT_BASE = flags.DEFINE_integer(
    "exponent_base", 10, "Base of exponential row number perturbations.")
_EXPONENT_RANGE = flags.DEFINE_integer(
    "exponent_range", 3, "One-sided range of exponent of perturbations.")
_MAX_PLANS_PER_GENERATION = flags.DEFINE_integer(
    "max_plans_per_generation", 20,
    "Max number of plans to mutate per generation.")
_PERTURB_UNIT_ONLY = flags.DEFINE_bool(
    "perturb_unit_only", True,
    "Whether to perturb only row counts exactly equal to one."
)
_MAX_PERTURBS_PER_JOIN = flags.DEFINE_integer(
    "max_perturbs_per_join", 1,
    "Limit on how many times a specific join can be perturbed."
)
# Exhaustive cardinality perturbation flags.
_CARDINALITY_MULTIPLIERS = flags.DEFINE_list(
    "cardinality_multipliers", None,
    "List of cardinality multipliers to apply when generating plans.")


def main(unused_argv):
  configs = _CONFIG_STR.value.split(",") if _CONFIG_STR.value else []

  with open(_QUERY_PARAMS_FILE.value) as json_file:
    info = json.load(json_file)

  hints_output_dir = os.path.join(_OUTPUT_DIR.value, _DATABASE.value)
  os.makedirs(hints_output_dir, exist_ok=True)

  database_configuration = query_utils.DatabaseConfiguration(
      dbname=_DATABASE.value,
      user=_USER.value,
      password=_PASSWORD.value,
      host=_HOST.value)
  query_manager = query_utils.QueryManager(database_configuration)
  query_utils.save_postgres_config_info(query_manager, _OUTPUT_DIR.value)

  hint_accumulator = main_utils.HintAccumulator()
  for query_id, query_metadata in info.items():
    logging.info("Start: %s", query_id)

    output = {}
    output["output"] = {}

    function_kwargs = {
        "database_configuration": database_configuration,
        "query": query_metadata["query"],
        "keys_to_remove": _KEYS_TO_REMOVE.value
    }

    # Augment kwargs depending on generation function.
    if _GENERATION_FUNCTION.value == GenerationFunction.PG_CONFIGS:
      function_kwargs["configs"] = configs
    elif _GENERATION_FUNCTION.value == GenerationFunction.ROW_NUM_EVOLUTION:
      function_kwargs.update({
          "max_plans": _MAX_PLANS_PER_PARAM.value,
          "num_generations": _NUM_GENERATIONS.value,
          "num_mutations_per_plan": _NUM_MUTATIONS_PER_PLAN.value,
          "exponent_base": _EXPONENT_BASE.value,
          "exponent_range": _EXPONENT_RANGE.value,
          "max_plans_per_generation": _MAX_PLANS_PER_GENERATION.value,
          "perturb_unit_only": _PERTURB_UNIT_ONLY.value,
          "max_perturbs_per_join": _MAX_PERTURBS_PER_JOIN.value
      })
    elif _GENERATION_FUNCTION.value == GenerationFunction.EXHAUSTIVE_CARDINALITY_PERTURBATIONS:
      cardinality_multipliers = [
          float(multiplier) for multiplier in _CARDINALITY_MULTIPLIERS.value
      ]

      function_kwargs.update(
          {"cardinality_multipliers": cardinality_multipliers})

    if _PARAMS_LIMIT.value:
      query_metadata["params"] = query_metadata["params"][:_PARAMS_LIMIT.value]

    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_generate_plan_candidates.execute_plan_generation(
        _GENERATION_FUNCTION_MAP[_GENERATION_FUNCTION.value],
        function_kwargs,
        query_metadata["params"],
        plan_hint_extractor=plan_hint_extractor,
        chunksize=_CHUNKSIZE.value,
        distributed=_supports_distributed_execution(_GENERATION_FUNCTION.value),
        soft_total_plans_limit=_SOFT_TOTAL_PLANS_LIMIT.value)
    counts, plan_hints, params_plan_indices, debug_infos = (
        plan_hint_extractor.get_consolidated_plan_hints())

    hint_accumulator.query_id_to_counts[query_id] = counts
    hint_accumulator.query_id_to_plan_hints[query_id] = plan_hints
    hint_accumulator.query_id_to_params_plan_indices[
        query_id] = params_plan_indices
    hint_accumulator.query_id_to_debug_infos[query_id] = debug_infos

    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=query_id,
        query=query_metadata["query"],
        plan_hints=plan_hints,
        params_plan_indices=params_plan_indices,
        database_configuration=database_configuration)
    hint_accumulator.combined_failure_counts.update(failure_counts)

  main_utils.print_failure_counts(hint_accumulator.combined_failure_counts)
  main_utils.print_hint_counts_by_source(hint_accumulator.query_id_to_counts)

  hint_accumulator.save(
      output_dir=hints_output_dir,
      plans_output_file=_PLANS_OUTPUT_FILE.value,
      verification_failures_file=_VERIFICATION_FAILURES_FILE.value,
      plan_index_suffix=_PLAN_INDEX_SUFFIX.value)

if __name__ == "__main__":
  app.run(main)

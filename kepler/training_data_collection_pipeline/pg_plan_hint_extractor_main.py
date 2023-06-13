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

"""Script to extract plan hints from explain plans.

Usage cases:
  1. Extract hints for a single query template.
  2. Extract hints for all queries in a provided file, and run a sanity-check
  verification on the extracted hints.
  3. Run only the verification on already-extracted hints.
  4. Merge hints from separate hint files together.
"""

import glob
import json
import os
import shutil

from absl import app
from absl import flags
from absl import logging

from kepler.training_data_collection_pipeline import main_utils
from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")
_HOST = flags.DEFINE_string("host", "localhost", "Database host.")

# 1. Extract from a single query plan file.
_SINGLE_HINT_FILE = flags.DEFINE_string(
    "single_hint_file", None,
    "Input file containing single query to extract hints for.")

# 2. Extract and verify for all files in a directory.
_INPUT_FILES_DIR = flags.DEFINE_string(
    "input_files_dir", None, "Directory in which query plans are stored.")
_INPUT_PREFIX = flags.DEFINE_string(
    "input_prefix", "query_plans_",
    "Prefix of files in which query plans are stored.")
_QUERY_TEMPLATES_FILE = flags.DEFINE_string(
    "query_templates_file", None,
    "Path to file in which query templates are stored.")
_PLANS_OUTPUT_FILE = flags.DEFINE_string(
    "plans_output_file", None,
    "File to store distinct plans per query. The file name is expected to end in .json"
)
_PLAN_INDEX_SUFFIX = flags.DEFINE_string(
    "plan_index_suffix", "_plan_index.json",
    "Suffix of files to store plan indices in.")
_VERIFICATION_FAILURES_FILE = flags.DEFINE_string(
    "verification_failures_file", "verification_failures.json",
    "Filename of file to save verification failures.")

# 3. Perform verification only.
_HINTS_TO_VERIFY_FILE = flags.DEFINE_string(
    "hints_to_verify_file", None, "Verify hints in the provided file.")
_VERIFICATION_OUTPUT_DIR = flags.DEFINE_string(
    "verification_output_dir", None,
    "Directory to populate when verification succeeds.")

# 4. Merge additional hints.
_ADDITIONAL_HINT_DIR = flags.DEFINE_string(
    "additional_hint_dir", None,
    "Directory containing per-query additional hints to merge.")
_ADDITIONAL_HINT_PREFIX = flags.DEFINE_string(
    "additional_hint_prefix", None,
    "Prefix of files containing additional hints (files will be of the form {prefix}_q{a}_{b}.json)."
)
_BASE_HINT_DIR = flags.DEFINE_string("base_hint_dir", None,
                                     "Directory containing base set of hints.")
_MERGE_SUFFIX = flags.DEFINE_string(
    "merge_suffix", "", "String to append to end of source of merged hints.")
_MERGE_DIR_SUFFIX = flags.DEFINE_string(
    "merge_dir_suffix", "_merged",
    "String to append to end of merged directory name")

_QUERY_IDS_TO_SKIP = ["q3_0", "q3_1", "q3_2"]


def _extract_single_query_hints(
    database_configuration: query_utils.DatabaseConfiguration):
  query = " ".join(
      pg_plan_hint_extractor.get_file_content(_SINGLE_HINT_FILE.value))
  hints = pg_plan_hint_extractor.get_single_query_hints(
      database_configuration=database_configuration, query=query)
  logging.info(hints)


def _load_queries():
  assert _QUERY_TEMPLATES_FILE.value, ("Query template file required.")
  queries = {}
  with open(_QUERY_TEMPLATES_FILE.value) as query_file:
    info = json.load(query_file)
    for query_id, query_metadata in info.items():
      queries[query_id] = query_metadata["query"]
  return queries


def _verify_provided_hints(
    database_configuration: query_utils.DatabaseConfiguration):
  """Verifies already extracted hints in _HINTS_TO_VERIFY_FILE.

  Writes output to _VERIFICATION_OUTPUT_DIR as ad-hoc signal that
  verification succeeded when parallelizing over many machines.

  Args:
    database_configuration: The configuration describing the database
      connection.
  """
  queries = _load_queries()

  with open(_HINTS_TO_VERIFY_FILE.value) as f:
    hints = json.load(f)

  with open(_HINTS_TO_VERIFY_FILE.value[:-5] + _PLAN_INDEX_SUFFIX.value) as f:
    plan_index = json.load(f)

  combined_failure_counts = {}
  for query_id in hints:
    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=query_id,
        query=queries[query_id],
        plan_hints=hints[query_id],
        params_plan_indices=plan_index[query_id],
        database_configuration=database_configuration)
    combined_failure_counts.update(failure_counts)

  main_utils.print_failure_counts(combined_failure_counts)

  filename = os.path.basename(_HINTS_TO_VERIFY_FILE.value)
  os.makedirs(_VERIFICATION_OUTPUT_DIR.value, exist_ok=True)
  with open(os.path.join(_VERIFICATION_OUTPUT_DIR.value, filename), "w") as f:
    f.write(filename)


def _extract_hints_and_verify(
    database_configuration: query_utils.DatabaseConfiguration):
  """Extract hints from file, and verify them using sanity check.

  Takes as input files from _INPUT_FILES_DIR, and writes two types of files
  to output:
    plan hints in the base _PLANS_OUTPUT_FILE
    plan indices in _PLANS_OUTPUT_FILE + _PLAN_INDEX_SUFFIX

  Args:
    database_configuration: The configuration describing the database
      connection.
  """
  queries = _load_queries()

  query_id_to_plans = {}
  filepaths = glob.glob(
      os.path.join(_INPUT_FILES_DIR.value, f"{_INPUT_PREFIX.value}*"))
  for filepath in filepaths:
    # This is brittle to the current usage and convention, but also much
    # faster than loading the plans file first.
    # Requires files to end in q[x]_[y].json.
    query_id = "_".join((os.path.basename(filepath)[:-5]).split("_")[-2:])
    with open(filepath) as json_file:
      info = json.load(json_file)
      query_id_to_plans[query_id] = info["output"]

  hint_accumulator = main_utils.HintAccumulator()

  for query_id, params_plans_info_list in query_id_to_plans.items():
    if query_id in _QUERY_IDS_TO_SKIP:
      continue
    plan_hint_extractor = pg_plan_hint_extractor.PlanHintExtractor()
    pg_plan_hint_extractor.add_query_plans_bulk(plan_hint_extractor,
                                                params_plans_info_list)
    counts, plan_hints, params_plan_indices, debug_infos = (
        plan_hint_extractor.get_consolidated_plan_hints())

    hint_accumulator.query_id_to_counts[query_id] = counts
    hint_accumulator.query_id_to_plan_hints[query_id] = plan_hints
    hint_accumulator.query_id_to_params_plan_indices[
        query_id] = params_plan_indices
    hint_accumulator.query_id_to_debug_infos[query_id] = debug_infos

    failure_counts = pg_plan_hint_extractor.verify_hints(
        query_id=query_id,
        query=queries[query_id],
        plan_hints=plan_hints,
        params_plan_indices=params_plan_indices,
        database_configuration=database_configuration)
    hint_accumulator.combined_failure_counts.update(failure_counts)

  main_utils.print_failure_counts(hint_accumulator.combined_failure_counts)
  main_utils.print_hint_counts_by_source(hint_accumulator.query_id_to_counts)

  plans_output_file_directory = os.path.dirname(_PLANS_OUTPUT_FILE.value)
  os.makedirs(plans_output_file_directory, exist_ok=True)
  hint_accumulator.save(
      output_dir=plans_output_file_directory,
      plans_output_file=_PLANS_OUTPUT_FILE.value,
      verification_failures_file=_VERIFICATION_FAILURES_FILE.value,
      plan_index_suffix=_PLAN_INDEX_SUFFIX.value)


def _merge_additional_hints():
  """Merge additional hints into set of base hints.

  Also writes a file containing what hints were new.

  Base hints and additional hints are outputs of extraction step and are
  assumed to be per-query (for distributed execution purposes), and are
  found in _BASE_HINT_DIR and _ADDITIONAL_HINT_DIR respectively.
  """
  outdir = _BASE_HINT_DIR.value + _MERGE_DIR_SUFFIX.value
  if os.path.exists(outdir):
    shutil.rmtree(outdir)

  os.mkdir(outdir)
  for filename in os.listdir(_BASE_HINT_DIR.value):
    filepath = os.path.join(_BASE_HINT_DIR.value, filename)
    out_filepath = os.path.join(outdir, filename)
    if os.path.isdir(filepath):
      continue
    if filename.endswith(_PLAN_INDEX_SUFFIX.value):
      shutil.copyfile(filepath, out_filepath)
      continue

    with open(filepath) as f:
      base_hints = json.load(f)

    # Typically there will only be a single query_id here.
    for query_id, hints in base_hints.items():
      additional_filename = f"{_ADDITIONAL_HINT_PREFIX.value}_{query_id}.json"
      with open(os.path.join(_ADDITIONAL_HINT_DIR.value,
                             additional_filename)) as additional_hints_file:
        additional_data = json.load(additional_hints_file)
        hints_to_merge = additional_data[query_id]
        num_hints_before = len(hints)
        pg_plan_hint_extractor.merge_hints(hints, hints_to_merge,
                                           _MERGE_SUFFIX.value)
        logging.info(
            "Query %s num hints before: %s; num additional %s; merged total: %s",
            query_id, num_hints_before, len(hints_to_merge), len(hints))

    with open(out_filepath, "w") as f:
      json.dump(base_hints, f)


def main(unused_argv):
  database_configuration = query_utils.DatabaseConfiguration(
      dbname=_DATABASE.value,
      user=_USER.value,
      password=_PASSWORD.value,
      host=_HOST.value)
  if _SINGLE_HINT_FILE.value:
    _extract_single_query_hints(database_configuration)
    return

  if _HINTS_TO_VERIFY_FILE.value:
    _verify_provided_hints(database_configuration)
    return
  elif _INPUT_FILES_DIR.value:
    _extract_hints_and_verify(database_configuration)
    return

  if _ADDITIONAL_HINT_DIR.value and _ADDITIONAL_HINT_PREFIX.value and _BASE_HINT_DIR.value:
    _merge_additional_hints()
    return


if __name__ == "__main__":
  app.run(main)

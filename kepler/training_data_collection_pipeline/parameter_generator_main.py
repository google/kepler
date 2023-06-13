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

"""Generate parameter bindings for the provided templates.

The parameter bindings for each template are saved to a different file as
indicated by the output-related flags.
"""
import collections
from concurrent import futures
import functools
import json
import logging
import os
import time

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import parameter_generator
from kepler.training_data_collection_pipeline import query_utils

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")
_HOST = flags.DEFINE_string("host", "localhost", "Database host.")
_SEED = flags.DEFINE_float("seed", 0, "Database random number seed.")

_TEMPLATE_FILE = flags.DEFINE_string("template_file", None,
                                     "Parameterized query template file.")
flags.mark_flag_as_required("template_file")
_COUNT = flags.DEFINE_integer(
    "count", 1000000, "The max number of parameters to generate per query.")

_COUNTS_OUTPUT_FILE = flags.DEFINE_string(
    "counts_output_file", None,
    "Output file to store the parameter counts per query.")
flags.mark_flag_as_required("counts_output_file")
_PARAMETERS_OUTPUT_DIR = flags.DEFINE_string(
    "parameters_output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("parameters_output_dir")

_DRY_RUN = flags.DEFINE_bool(
    "dry_run", False,
    "If true, verify that the parameter generation process works correctly "
    "using a single, non-random parameter value. This involves a) verifying "
    "that the parameter generation query can be composed from the template "
    "query and b) ensuring that the template query executes successfully with "
    "the generated parameter value.")


def main(unused_argv):
  with open(_TEMPLATE_FILE.value) as f:
    templates = json.load(f)

  work_list = []
  # Query templates that failed hint verification using the parameters from the
  # original Stack benchmark. That is, at least one provided hint was ignored by
  # the PG optimizer for at least one parameter binding.
  skip_list = ["q3_0", "q3_1", "q3_2"]
  for query_id, template in templates.items():
    if query_id not in skip_list:
      work_list.append(
          parameter_generator.TemplateItem(
              query_id=query_id, template=template))

  database_configuration = query_utils.DatabaseConfiguration(
      dbname=_DATABASE.value,
      user=_USER.value,
      password=_PASSWORD.value,
      host=_HOST.value,
      seed=_SEED.value)
  generator = parameter_generator.ParameterGenerator(database_configuration)
  parameter_generation_function = functools.partial(
      generator.generate_parameters, _COUNT.value, dry_run=_DRY_RUN.value)

  output_counts = collections.defaultdict(lambda: {})
  # The high-latency work occurs remotely via the database executing queries to
  # generate parameters. The number of max workers is limited empirically to
  # avoid memory issues on the database side.
  with futures.ThreadPoolExecutor(max_workers=3) as executor:
    for result in executor.map(parameter_generation_function, work_list):
      query_id = next(iter(result))
      logging.info("Finished generating for %s", query_id)
      with open(
          os.path.join(_PARAMETERS_OUTPUT_DIR.value,
                       f"{query_id}-{len(result[query_id]['params'])}.json"),
          "w") as f:
        json.dump(result, f)

      output_counts[query_id] = len(result[query_id]["params"])

      if _DRY_RUN.value:
        # Ensure that the template query executes successfully with the
        # generated parameter value.
        query_manager = query_utils.QueryManager(database_configuration)
        start_ms = int(time.time() * 1e3)
        query = result[query_id]["query"]
        params = result[query_id]["params"][0]
        results = query_manager.execute(query, params)
        end_ms = int(time.time() * 1e3)

        print(f"Query {query_id} approximate latency: {end_ms-start_ms} ms")
        print(f"Template: {query}")
        print(f"Params: {params}")

        # The query should return at least one result.
        assert results

  with open(_COUNTS_OUTPUT_FILE.value, "w") as f:
    json.dump(output_counts, f)


if __name__ == "__main__":
  app.run(main)

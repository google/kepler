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

"""Outputs instances of query templates populated with bind parameters."""
import json

from absl import app
from absl import flags
from tensorflow.io import gfile

from kepler.training_data_collection_pipeline import query_text_utils

_PARAMETER_VALUES_FILE = flags.DEFINE_string("parameter_values_file", None,
                                             "Parameter values file.")
flags.mark_flag_as_required("parameter_values_file")

_LIMIT = flags.DEFINE_integer("limit", 100,
                              "Limit the number of parameters to instantiate.")

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None,
    "File to append the instantiated queries. If no file is provided, the "
    "instantiated queries will be printed to the console instead.")


def main(unused_argv):
  with gfile.GFile(_PARAMETER_VALUES_FILE.value) as f:
    parameter_values = json.load(f)

  if len(parameter_values) != 1:
    raise ValueError(
        "The parameter_values_file should contain contents for a single query"
        f"id, but found the follow query ids instead: {parameter_values.keys()}"
    )

  query_id = next(iter(parameter_values))
  query_info = parameter_values[query_id]

  query_instances = [
      query_text_utils.substitute_query_params(query_info["query"], params)
      for params in query_info["params"][:_LIMIT.value]
  ]
  query_instances = [
      query_instance +
      ";" if query_instance.strip()[-1] != ";" else query_instance
      for query_instance in query_instances
  ]

  description = f"-- Queries for {query_id}"
  if _OUTPUT_FILE.value:
    with gfile.GFile(_OUTPUT_FILE.value, "a") as f:
      f.write(description + "\n")
      for query_instance in query_instances:
        f.write(f"{query_instance}\n\n")

  else:
    for query_instance in query_instances:
      print(f"{query_instance}\n")


if __name__ == "__main__":
  app.run(main)

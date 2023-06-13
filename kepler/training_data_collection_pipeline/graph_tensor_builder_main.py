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

"""Converts Postgres query EXPLAIN plans to TF GNN GraphTensor.

Graph tensors are encoded as tf.train.Example per recommendation in:
https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md
"""

import json
import os
from typing import List

from absl import app
from absl import flags

import tensorflow as tf

from kepler.training_data_collection_pipeline import graph_tensor_builder
from kepler.training_data_collection_pipeline import query_utils

_DATABASE = flags.DEFINE_string("database", None, "Database name.")
flags.mark_flag_as_required("database")
_USER = flags.DEFINE_string("user", None, "Database username.")
_PASSWORD = flags.DEFINE_string("password", None, "Database password.")
_HOST = flags.DEFINE_string("host", "localhost", "Database host.")
_SEED = flags.DEFINE_float("seed", 0, "Database random number seed.")

_QUERY = flags.DEFINE_string("query", None, "Specific query id to execute.")
flags.mark_flag_as_required("query")

_GRAPH_SCHEMA_FILENAME = flags.DEFINE_string(
    "graph_schema_filename", None,
    "The path to the file containing the graph schema.")
flags.mark_flag_as_required("graph_schema_filename")

_QUERY_TEMPLATES_FILE = flags.DEFINE_string(
    "query_templates_file", None,
    "Path to file in which query templates are stored.")
flags.mark_flag_as_required("query_templates_file")
_PLAN_HINTS_FILE = flags.DEFINE_string("plan_hints_file", None,
                                       "Plan hints file.")
flags.mark_flag_as_required("plan_hints_file")

_QUERY_EXECUTION_DATA_FILE = flags.DEFINE_string(
    "query_execution_data_file", None,
    "Path to file containing execution data for a given query.")
flags.mark_flag_as_required("query_execution_data_file")
_QUERY_EXECUTION_METADATA_FILE = flags.DEFINE_string(
    "query_execution_metadata_file", None,
    "Path to file containing execution metadata for a given query.")
flags.mark_flag_as_required("query_execution_metadata_file")

_LIMIT = flags.DEFINE_integer(
    "limit", None,
    "Limit the number of parameters per query to gather execution data for.")

_OUTPUT_DIR = flags.DEFINE_string("output_dir", None,
                                  "Directory to store tf.train.Examples.")
flags.mark_flag_as_required("output_dir")


def main(unused_argv):
  database_configuration = query_utils.DatabaseConfiguration(
      dbname=_DATABASE.value,
      user=_USER.value,
      password=_PASSWORD.value,
      host=_HOST.value,
      seed=_SEED.value)

  query_id = _QUERY.value

  with open(_QUERY_TEMPLATES_FILE.value) as f:
    templates = json.load(f)

  with open(_PLAN_HINTS_FILE.value) as f:
    plan_hints = json.load(f)

  with open(_QUERY_EXECUTION_DATA_FILE.value) as f:
    query_execution_data = json.load(f)

  with open(_QUERY_EXECUTION_METADATA_FILE.value) as f:
    query_execution_metadata = json.load(f)

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  def write_examples(graph_tensor_examples: List[str], plan_id: int) -> None:
    outfile = os.path.join(_OUTPUT_DIR.value,
                           f"{query_id}_hint_{plan_id}_examples")

    with tf.io.TFRecordWriter(outfile) as f:
      for example in graph_tensor_examples:
        f.write(example)

  graph_tensor_builder.create_examples(
      database_configuration=database_configuration,
      graph_schema_filename=_GRAPH_SCHEMA_FILENAME.value,
      query_id=query_id,
      templates=templates,
      query_execution_data=query_execution_data,
      query_execution_metadata=query_execution_metadata,
      plan_hints=plan_hints,
      write_examples_fn=write_examples,
      limit=_LIMIT.value)


if __name__ == "__main__":
  app.run(main)

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

"""Runs a Kepler model prediction server.

The ModelServer supports any predictor implementing ModelPredictorBase. This
program instantiates a ModelServer using an
sngp_multihead_model.SNGPMultiheadModel converted to tflite.
"""

import json
import os

from absl import app
from absl import flags

from kepler.database_integrations.model_serving import model_server
from kepler.database_integrations.model_serving import query_parsing_utils
from kepler.model_trainer import sngp_multihead_model

_LOCAL_HOST = "127.0.0.1"
_KEPLER_PORT = 30709
_TFLITE_FILE_ENDING = ".tflite"

_HOST = flags.DEFINE_string(
    "host", _LOCAL_HOST, "The host address for the model server."
)
_PORT = flags.DEFINE_integer(
    "port", _KEPLER_PORT, "The port for the model server to use."
)

_QUERY_TEMPLATES_FILE = flags.DEFINE_string(
    "query_templates_file",
    None,
    "Path to file in which query templates are stored.",
)
flags.mark_flag_as_required("query_templates_file")

_PLAN_HINTS_DIR = flags.DEFINE_string(
    "plan_hints_dir",
    None,
    (
        "Directory containing plan hints files. The directory must contain a"
        " hint file for each query id found in _TFLITE_SNGP_MODEL_DIR. Hint"
        " files for additional query ids will not cause a problem."
    ),
)
flags.mark_flag_as_required("plan_hints_dir")

_PLAN_COVERS_DIR = flags.DEFINE_string(
    "plan_covers_dir",
    None,
    (
        "Directory containing the metadata files which contain the plan_cover."
        " These are generated during training data collection. The directory"
        " must contain a metadata file for each query id found in"
        " _TFLITE_SNGP_MODEL_DIR. Metadata files for additional query ids will"
        " not cause a problem."
    ),
)
flags.mark_flag_as_required("plan_covers_dir")

_TFLITE_SNGP_MODEL_DIR = flags.DEFINE_string(
    "tflite_sngp_model_dir",
    None,
    (
        "Directory containing tflite SNGP models to use for prediction. Assumes"
        " the contents of the directory are files named <query_id>.tflite"
    ),
)
flags.mark_flag_as_required("tflite_sngp_model_dir")


_QUERY_METADATA_FILE = flags.DEFINE_string(
    "query_metadata_file",
    None,
    (
        "File containing metadata describing the query predicates. This is the"
        " same file used in training."
    ),
)
flags.mark_flag_as_required("query_metadata_file")

_CONFIDENCE_THRESHOLD = flags.DEFINE_float(
    "confidence_threshold",
    0.9,
    (
        "The confidence threshold used by the SNGP models to determine whether"
        " to a prediction or abstain."
    ),
)


def main(unused_argv):
  with open(_QUERY_TEMPLATES_FILE.value) as f:
    templates = json.load(f)

  param_extractors = {}
  for query_id, template_entry in templates.items():
    param_extractors[query_id] = query_parsing_utils.ParamExtractor(
        query_template=template_entry["query"]
    )

  plan_hints = {}
  for plan_hints_file in os.listdir(_PLAN_HINTS_DIR.value):
    with open(os.path.join(_PLAN_HINTS_DIR.value, plan_hints_file)) as f:
      query_plan_hints = json.load(f)

    query_id_overlap = set(query_plan_hints).intersection(set(plan_hints))
    assert not query_id_overlap, (
        "One of these query ids was observed in multiple hint files:"
        f" {query_id_overlap}"
    )
    plan_hints.update(query_plan_hints)

  plan_covers = {}
  for plan_cover_file in os.listdir(_PLAN_COVERS_DIR.value):
    with open(os.path.join(_PLAN_COVERS_DIR.value, plan_cover_file)) as f:
      metadata = json.load(f)

    query_id_overlap = set(metadata).intersection(set(plan_covers))
    assert not query_id_overlap, (
        "One of these query ids was observed in multiple metadata files:"
        f" {query_id_overlap}"
    )
    for query_id, entry in metadata.items():
      plan_covers[query_id] = entry["plan_cover"]

  with open(_QUERY_METADATA_FILE.value) as f:
    query_metadata = json.load(f)

  tflite_predictor_files = os.listdir(_TFLITE_SNGP_MODEL_DIR.value)
  predictors = {}
  for tflite_predictor_file in tflite_predictor_files:
    query_id = tflite_predictor_file[
        : tflite_predictor_file.index(_TFLITE_FILE_ENDING)
    ]

    with open(
        os.path.join(_TFLITE_SNGP_MODEL_DIR.value, tflite_predictor_file), "rb"
    ) as f:
      tflite_model_content = f.read()
    predictor = sngp_multihead_model.SNGPMultiheadModelTFLitePredictor(
        tflite_model_content=tflite_model_content,
        metadata=query_metadata[query_id],
        plan_cover=plan_covers[query_id],
        confidence_threshold=_CONFIDENCE_THRESHOLD.value,
    )
    predictors[query_id] = predictor

  server = model_server.ModelServer(
      host=_HOST.value,
      port=_PORT.value,
      param_extractors=param_extractors,
      plan_hints=plan_hints,
      predictors=predictors,
  )
  server.run()


if __name__ == "__main__":
  app.run(main)

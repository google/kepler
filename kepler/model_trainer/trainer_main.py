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

"""Example script for training a Kepler model.

This script is mostly designed for illustrative purposes, hence it does not
fully parameterize things like model type, model architectures, hyperparameters,
etc. We recommend using Colab for model experimentation instead.
"""
import json
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from kepler.data_management import database_simulator
from kepler.data_management import workload
from kepler.model_trainer import evaluation
from kepler.model_trainer import model_base
from kepler.model_trainer import sngp_multihead_model
from kepler.model_trainer import trainer
from kepler.model_trainer import trainer_util

_QUERY_METADATA = flags.DEFINE_string(
    "query_metadata",
    None,
    (
        "File containing query metadata which includes information on parameter"
        " values."
    ),
)
flags.mark_flag_as_required("query_metadata")

_EXECUTION_DATA = flags.DEFINE_string(
    "execution_data",
    None,
    "File containing query execution data across various parameter values.",
)
flags.mark_flag_as_required("execution_data")

_EXECUTION_METADATA = flags.DEFINE_string(
    "execution_metadata",
    None,
    "File containing query execution metadata across various parameter values.",
)
flags.mark_flag_as_required("execution_metadata")

_PREPROCESSING_CONFIG = flags.DEFINE_string(
    "preprocessing_config",
    None,
    "File containing preprocessing config information.",
)

_VOCAB_DATA_DIR = flags.DEFINE_string(
    "vocab_data_dir", None, "Folder containing distinct values for each column."
)
flags.mark_flag_as_required("vocab_data_dir")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_SEED = flags.DEFINE_integer(
    "seed", 0, "Seed used to shuffle workload before splitting."
)

_NUM_TRAINING_PARAMETERS = flags.DEFINE_integer(
    "num_training_parameters",
    10,
    "Number of parameters (and related plans) to use for training.",
)
_NUM_EVALUATION_PARAMETERS = flags.DEFINE_integer(
    "num_evaluation_parameters",
    10,
    "Number of parameters (and related plans) to evaluate on.",
)
_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 20, "Number of epochs to train for."
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Training minibatch size.")
_VOCAB_SIZE_LIMIT = flags.DEFINE_integer(
    "vocab_size_limit", 200, "Maximum vocabulary size."
)


def _get_distinct_values(table, column):
  with open(
      os.path.join(_VOCAB_DATA_DIR.value, f"{table}-{column}-distinct_values")
  ) as f:
    return json.load(f)


def main(unused_argv):
  query_id = _QUERY_ID.value
  with open(_QUERY_METADATA.value) as json_file:
    query_metadata = json.load(json_file)
    metadata = query_metadata[query_id]

  with open(_EXECUTION_DATA.value) as json_file:
    execution_data = json.load(json_file)

  with open(_EXECUTION_METADATA.value) as json_file:
    execution_metadata = json.load(json_file)

  if _PREPROCESSING_CONFIG.value:
    with open(_PREPROCESSING_CONFIG.value) as json_file:
      preprocessing_config = json.load(json_file)
  else:
    preprocessing_config = trainer_util.construct_preprocessing_config(
        metadata)

  plans = workload.KeplerPlanDiscoverer(
      query_execution_metadata=execution_metadata)
  database = database_simulator.DatabaseSimulator(
      query_execution_data=execution_data,
      query_execution_metadata=execution_metadata,
      estimator=database_simulator.LatencyEstimator.MIN,
  )
  client = database_simulator.DatabaseClient(database)
  workload_generator = workload.WorkloadGenerator(execution_data)
  full_workload = workload_generator.all()
  workload.shuffle(full_workload, seed=_SEED.value)

  workload_train, workload_eval = workload.split(
      full_workload, first_half_count=_NUM_TRAINING_PARAMETERS.value
  )

  # Additionally fetch all default execution times to compute near-optimality.
  queries_train = workload.create_query_batch(plans.plan_ids, workload_train)
  queries_train_with_default = workload.create_query_batch(
      plans.plan_ids + [None], workload_train)
  query_execution_train_df = client.execute_timed_batch(
      planned_queries=queries_train)
  query_execution_train_with_default_df = client.execute_timed_batch(
      planned_queries=queries_train_with_default)

  default_latencies_train = evaluation.get_default_latencies(
      database=database,
      query_workload=workload_train)
  default_latencies_train = np.atleast_2d(default_latencies_train).T

  trainer_util.add_vocabulary_to_metadata(
      query_execution_train_df, metadata,
      _get_distinct_values, _VOCAB_SIZE_LIMIT.value)

  logging.info(
      "Num queries: %d, num plans %d",
      len(queries_train),
      len(plans.plan_ids),
  )

  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model_config = model_base.ModelConfig(
      layer_sizes=[64, 64, 64],
      dropout_rates=[0., 0., 0.],
      learning_rate=3e-4,
      activation="relu",
      spectral_norm_multiplier=10.0,
      num_gp_random_features=128,
      loss=loss,
      metrics=[]
  )

  model = sngp_multihead_model.SNGPMultiheadModel(
      metadata, plans.plan_ids, model_config,
      preprocessing_config)

  t = trainer.NearOptimalClassificationTrainer(metadata, plans, model)
  x, y = t.construct_training_data(query_execution_train_with_default_df,
                                   default_relative=True,
                                   near_optimal_threshold=1.1)

  # Construct sample weight matrix.
  all_latencies = np.array(query_execution_train_df["latency_ms"]).reshape(
      (-1, len(plans.plan_ids)))
  sample_weight = trainer_util.get_sample_weight(all_latencies,
                                                 default_latencies_train)

  print("Training on %d samples" % len(y))
  t.train(x, y,
          epochs=_NUM_EPOCHS.value,
          batch_size=_BATCH_SIZE.value,
          sample_weight=sample_weight)

  # Perform some basic model evaluation.
  model_predictor = sngp_multihead_model.SNGPMultiheadModelPredictor(
      model.get_model(), metadata, plan_cover=plans.plan_ids,
      confidence_threshold=0.9)

  eval_inputs = trainer_util.construct_multihead_model_inputs(workload_eval)
  plan_selections, _ = model_predictor.predict(eval_inputs)

  candidate_latencies = evaluation.get_candidate_latencies(
      database=database,
      query_workload=workload_eval,
      plan_selections=plan_selections)
  default_latencies = evaluation.get_default_latencies(
      database=database,
      query_workload=workload_eval)
  optimal_latencies = evaluation.get_optimal_latencies(
      client=client,
      query_workload=workload_eval,
      kepler_plan_discoverer=plans)

  evaluation.evaluate(
      candidate_latencies=candidate_latencies,
      default_latencies=default_latencies,
      optimal_latencies=optimal_latencies)


if __name__ == "__main__":
  app.run(main)

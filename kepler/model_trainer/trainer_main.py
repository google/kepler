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

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from kepler.data_management import database_simulator
from kepler.data_management import workload
from kepler.model_trainer import evaluation
from kepler.model_trainer import model_base
from kepler.model_trainer import multihead_model
from kepler.model_trainer import trainer
from kepler.model_trainer import trainer_util

_QUERY_METADATA = flags.DEFINE_string(
    "query_metadata", None,
    "File containing query metadata which includes information on parameter values."
)
flags.mark_flag_as_required("query_metadata")

_EXECUTION_DATA = flags.DEFINE_string(
    "execution_data", None,
    "File containing query execution data across various parameter values.")
flags.mark_flag_as_required("execution_data")

_EXECUTION_METADATA = flags.DEFINE_string(
    "execution_metadata", None,
    "File containing query execution metadata across various parameter values.")
flags.mark_flag_as_required("execution_metadata")

_PREPROCESSING_CONFIG = flags.DEFINE_string(
    "preprocessing_config", None,
    "File containing preprocessing config information.")
flags.mark_flag_as_required("execution_data")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_SEED = flags.DEFINE_integer("seed", 0,
                             "Seed used to shuffle workload before splitting.")

_NUM_TRAINING_PARAMETERS = flags.DEFINE_integer(
    "num_training_parameters", 10,
    "Number of parameters (and related plans) to use for training.")
_NUM_EVALUATION_PARAMETERS = flags.DEFINE_integer(
    "num_evaluation_parameters", 10,
    "Number of parameters (and related plans) to evaluate on.")
_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 20,
                                   "Number of epochs to train for.")


def main(unused_argv):
  with open(_QUERY_METADATA.value) as json_file:
    query_metadata = json.load(json_file)
    metadata = query_metadata[_QUERY_ID.value]

  with open(_EXECUTION_DATA.value) as json_file:
    execution_data = json.load(json_file)

  with open(_EXECUTION_METADATA.value) as json_file:
    execution_metadata = json.load(json_file)

  with open(_PREPROCESSING_CONFIG.value) as json_file:
    preprocessing_config = json.load(json_file)

  plans = workload.KeplerPlanDiscoverer(execution_data)
  database = database_simulator.DatabaseSimulator(
      query_execution_data=execution_data,
      query_execution_metadata=execution_metadata,
      estimator=database_simulator.LatencyEstimator.MIN)
  client = database_simulator.DatabaseClient(database)
  workload_generator = workload.WorkloadGenerator(execution_data)
  full_workload = workload_generator.all()
  workload.shuffle(full_workload, seed=_SEED.value)

  workload_train, workload_remaining = workload.split(
      full_workload, first_half_count=_NUM_TRAINING_PARAMETERS.value)
  queries = workload.create_query_batch(plans.plan_ids, workload_train)
  query_execution_df = client.execute_timed_batch(planned_queries=queries)

  logging.info("Num queries: %d, num plans %d, total num training points %d",
               len(queries), len(plans.plan_ids), len(query_execution_df))

  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  model_config = model_base.ModelConfig(
      layer_sizes=[128, 64],
      dropout_rates=[0.25, 0.25],
      learning_rate=1e-3,
      activation="relu",
      loss=loss,
      metrics=["accuracy",
               tf.keras.metrics.AUC(multi_label=True, from_logits=True)]
  )
  model = multihead_model.MultiheadModel(metadata, plans.plan_ids, model_config,
                                         preprocessing_config)

  classification_trainer = trainer.ClassificationTrainer(metadata,
                                                         plans, model)
  x, y = classification_trainer.construct_training_data(query_execution_df)

  classification_trainer.train(x, y, epochs=_NUM_EPOCHS.value, batch_size=64)

  workload_eval, workload_remaining = workload.split(
      workload_remaining, first_half_count=_NUM_EVALUATION_PARAMETERS.value)

  eval_inputs = trainer_util.construct_multihead_model_inputs(
      workload_eval, metadata["predicates"])
  plan_selections = model.get_model_predictions(eval_inputs)

  candidate_latencies = evaluation.get_candidate_latencies(
      database=database,
      query_workload=workload_eval,
      plan_selections=plan_selections)
  default_latencies = evaluation.get_default_latencies(
      database=database, query_workload=workload_eval)
  optimal_latencies = evaluation.get_optimal_latencies(
      client=client, query_workload=workload_eval, kepler_plan_discoverer=plans)

  evaluation.evaluate(
      candidate_latencies=candidate_latencies,
      default_latencies=default_latencies,
      optimal_latencies=optimal_latencies)


if __name__ == "__main__":
  app.run(main)

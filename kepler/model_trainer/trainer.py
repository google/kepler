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

"""Training framework for Kepler models.

This module implements a variety of trainer classes that encapsulate model
training in a variety of settings, including classification, regression, and
ranking scenarios.

These models are intended to be trained using query execution latencies
of a set of plans executed over a given workload.
"""

import abc
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from kepler.data_management import workload
from kepler.model_trainer import model_base
from kepler.model_trainer import trainer_util

JSON = Any


class TrainerBase(metaclass=abc.ABCMeta):
  """Base trainer class for Kepler modeling.

  The trainer manages model training over a fixed set of plans.  Once the
  workload execution latencies are collected, the trainer provides functionality
  to transform them into the proper training data format.
  """

  def __init__(self,
               metadata: JSON,
               plans: workload.KeplerPlanDiscoverer,
               model: model_base.ModelBase):
    """Initializes a TrainerBase.

    Args:
      metadata: Metadata for entire query (i.e. under query_id key). The
        predicates value is structured as a list, with each entry corresponding
        to a param. Each entry in this list is a dict containing data_type (int,
        float, text), and associated info, e.g. min/max ranges for ints,
        distinct values for embedded parameters, etc.
      plans: Specifies the set of candidate plans to use.
      model: The model to train.
    """
    self._predicate_metadata = metadata["predicates"]
    # Model might only predict a subset of all plans (e.g. plan cover).
    self._plan_id_to_index = {plan_id: i for i, plan_id in
                              enumerate(plans.plan_ids)}

    self._model = model

  def apply_preprocessing(self, execution_df: pd.DataFrame) -> None:
    """Applies preprocessing to parameter columns in-place.

    Args:
      execution_df: Dataframe containing training data (e.g. from
        DatabaseClient.execute_timed_batch).
    """
    for i, predicate in enumerate(self._predicate_metadata):
      col_name = trainer_util.get_parameter_column_name(i)
      if predicate.get("preprocess_type") == "to_timestamp":
        execution_df[col_name] = (
            pd.to_datetime(
                execution_df[col_name]).apply(lambda x: x.timestamp()))

  def train(
      self,
      x: List[Any],
      y: Any,
      epochs: int = 20,
      batch_size: int = 32,
      sample_weight: Optional[np.ndarray] = None) -> tf.keras.callbacks.History:
    """Trains the model.

    Args:
      x: Training data inputs.
      y: Training data targets.
      epochs: Number of epochs.
      batch_size: Training batch size.
      sample_weight: Weights loss for each training example.

    Returns:
      History object containing model training loss and metrics.
    """
    return self._model.get_model().fit(
        x=x,
        y=y,
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weight)

  def get_parameter_column_names(self) -> List[str]:
    """Get column names for each parameter.

    Returns:
      List containing all parameter names in order.
    """
    return trainer_util.get_parameter_column_names(
        parameter_count=len(self._predicate_metadata))

  @abc.abstractmethod
  def construct_training_data(
      self,
      query_execution_df: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray]:
    """Construct training data inputs/outputs for the particular trainer.

    Implementations of this method should try to avoid modifying
    query_execution_df (e.g. by making a copy).

    Args:
      query_execution_df: Dataframe containing training data (e.g. from
        DatabaseClient.execute_timed_batch).

    Returns:
      Tuple consisting of training data inputs and outputs.
    """
    raise NotImplementedError


class ClassificationTrainer(TrainerBase):
  """Trainer for predicting the single optimal plan.

  The target for any input is the index corresponding to the optimal plan,
  i.e. the plan with the lowest execution latency.
  """

  def construct_training_data(
      self,
      query_execution_df: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray]:
    """Gets training data inputs/outputs for classification.

    Args:
      query_execution_df: Dataframe containing training data (e.g. from
        DatabaseClient.execute_timed_batch).

    Returns:
      Tuple consisting of:
        1. Training data inputs: List of 1d arrays each containing all the
          values over the dataset for a single feature.
        2. Training data outputs: 2d one-hot array indicating which plan is
          optimal for each parameter.
    """
    # Get a copy without without unused columns.
    query_execution_df = query_execution_df.drop(["is_default", "total_cost"],
                                                 axis=1,
                                                 errors="ignore")

    # Apply preprocessing to parameter columns.
    self.apply_preprocessing(query_execution_df)

    # Get the list of column names and cast any required columns to the
    # appropriate type.
    column_names = self.get_parameter_column_names()
    query_execution_df = trainer_util.cast_df_columns(
        query_execution_df, self._predicate_metadata)

    # Get single row for each parameter corresponding to the fastest plan id.
    optimal_plans = query_execution_df.loc[query_execution_df.groupby(
        column_names, sort=False)["latency_ms"].idxmin()]
    optimal_plans = optimal_plans.drop("latency_ms", axis=1)
    optimal_plans = optimal_plans.rename(columns={"plan_id": "optimal_plan_id"})

    # Use one-hot targets.
    optimal_idxs = np.array(optimal_plans.pop("optimal_plan_id")).astype(int)
    target = np.zeros((len(optimal_idxs), len(self._plan_id_to_index)))
    target[np.arange(target.shape[0]),
           list(map(lambda x: self._plan_id_to_index[x], optimal_idxs))] = 1

    # Split the columns since they may be different types and need to be passed
    # into the model as multiple inputs.
    feature_values = [np.array(optimal_plans.pop(n))
                      for n in optimal_plans.columns]
    return feature_values, target


class PairwiseRankingTrainer(TrainerBase):
  """Trainer for performing pairwise ranking.

  Constructs training data for any pairwise ranking model by getting
  all possible unique pairs of plans, and the binary variable corresponding to
  which plan is faster.

  TODO(b/217974851): All pairs can be quite expensive for large number of
  plans. Either do plan reduction, or do sampling on the pairs.

  TODO(b/217974851): Handle timeouts properly. It doesn't make sense to
  compare two timed out plans, so filtering all timeout pairs is probably
  better.
  """

  def construct_training_data(
      self,
      query_execution_df: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray]:
    """Constructs dataset consisting of all pairs of plans.

    Args:
      query_execution_df: Dataframe containing training data (e.g. from
        DatabaseClient.execute_timed_batch).

    Returns:
      Tuple consisting of:
        1. Input features to pairwise ranking model. Standard input features
           (parameters and plan_id) are duplicated with _a, _b suffixes.
        2. Output labels; 1 if a is faster than b and 0 otherwise.
    """
    # Get a copy without without unused columns.
    query_execution_df = query_execution_df.drop(["is_default", "total_cost"],
                                                 axis=1,
                                                 errors="ignore")

    # Apply preprocessing to parameter columns.
    self.apply_preprocessing(query_execution_df)

    # Get the list of column names and cast any required columns to the
    # appropriate type.
    column_names = self.get_parameter_column_names()
    query_execution_df = trainer_util.cast_df_columns(
        query_execution_df, self._predicate_metadata)
    query_execution_df["plan_id"] = pd.to_numeric(query_execution_df["plan_id"])

    # Generate rows for the pair-wise model. We first group the data by
    # parameter values and then construct rows that compare the latency between
    # each plan.
    query_execution_df = pd.merge(query_execution_df,
                                  query_execution_df,
                                  on=column_names, suffixes=("_a", "_b"))
    query_execution_df.drop(
        query_execution_df[query_execution_df["plan_id_a"] >=
                           query_execution_df["plan_id_b"]].index,
        inplace=True)

    target = (query_execution_df["latency_ms_a"] >
              query_execution_df["latency_ms_b"]).astype(int)
    query_execution_df.drop(["latency_ms_a", "latency_ms_b"],
                            axis=1, inplace=True)

    column_names_a = [c + "_a" for c in column_names]
    column_names_b = [c + "_b" for c in column_names]
    query_execution_df.columns = [
        str(c) + "_a" if str(c) in column_names else c
        for c in query_execution_df.columns
    ]

    # Pairwise ranking model takes as input the following:
    # [input_features_a, plan_id_a, input_features_b, plan_id_b]
    # where input_features_a and input_features_b are identical.
    query_execution_df[column_names_b] = query_execution_df[column_names_a]
    order = query_execution_df.columns.tolist()
    order.remove("plan_id_b")
    order.append("plan_id_b")
    query_execution_df = query_execution_df[order]

    # Split the columns since they may be different types and need to be passed
    # into the model as multiple inputs.
    names = query_execution_df.columns.values.tolist()
    feature_values = [np.array(query_execution_df.pop(n)) for n in names]
    return feature_values, np.array(target)


class RegressionTrainer(TrainerBase):
  """Trainer for regressing against execution latencies."""

  def construct_training_data(
      self,
      query_execution_df: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray]:
    """Gets training data for regression against latencies.

    Args:
      query_execution_df: Dataframe containing training data (e.g. from
        DatabaseClient.execute_timed_batch).

    Returns:
      Tuple consisting of:
        1. Training data inputs: List of 1d arrays each containing all the
          values over the dataset for a single feature.
        2. Training data outputs: 2d array with shape (num_params, num_plans)
          containing all estimated execution latencies.
    """
    # Get a copy without without unused columns.
    query_execution_df = query_execution_df.drop(["is_default", "total_cost"],
                                                 axis=1,
                                                 errors="ignore")

    # Apply preprocessing to parameter columns.
    self.apply_preprocessing(query_execution_df)

    # Get the list of column names and cast any required columns to the
    # appropriate type.
    column_names = self.get_parameter_column_names()
    query_execution_df = trainer_util.cast_df_columns(
        query_execution_df, self._predicate_metadata)

    num_plans = len(self._plan_id_to_index)
    latencies = query_execution_df.pop("latency_ms")
    targets = np.array(latencies).reshape((-1, num_plans))

    # Get rid of any columns not corresponding to parameters.
    query_execution_df = query_execution_df.drop(
        set(query_execution_df.columns).difference(set(column_names)), axis=1)
    # Keep one row per parameter.
    query_execution_df.drop_duplicates(inplace=True)
    assert len(query_execution_df) == len(targets)

    # Split the columns since they may be different types and need to be passed
    # into the model as multiple inputs.
    feature_values = [np.array(query_execution_df.pop(n))
                      for n in query_execution_df.columns]
    return feature_values, targets


class NearOptimalClassificationTrainer(TrainerBase):
  """Trainer for predicting whether each plan is near-optimal.

  Each model output is head is trained using binary cross entropy loss.
  """

  def construct_training_data(
      self,
      query_execution_df: pd.DataFrame,
      near_optimal_threshold: float = 1.1,
      default_relative: bool = False,
  ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Gets training data for multi-label classification.

    All near-optimal plans have positive labels in this setting. The definition
    of near-optimal depends on default_relative. Let d be default plan's
    latency, o be optimal plan's latency, l be any candidate plan's latency,
    n be near_optimal_threshold. The candidate plan is near-optimal if:
      1. default_relative == True: d - o <= (d - l) * n
      2. default_relative == False: l <= o * n

    The default-relative objective may be more appropriate for the case when
    numerous plans are substantially faster than the default plan. Furthermore,
    if all plans in the plan cover are worse than the default, all their labels
    will be 0, i.e. this objective also implicitly avoids regressions.

    TODO(b/217974851): Assign positive labels to plans that are near-optimal to
    an optimal default plan. This may provide a slightly better signal.

    Args:
      query_execution_df: Dataframe containing training data (e.g. from
        DatabaseClient.execute_timed_batch).
      near_optimal_threshold: If latency * this value < optimal latency, we say
        this plan is near-optimal.
      default_relative: Controls the near-optimality definition (see above).
        If true, query_execution_df must contain data for all default plans.

    Returns:
      Tuple consisting of:
        1. Training data inputs: List of 1d arrays each containing all the
          values over the dataset for a single feature.
        2. Training data outputs: 2d array with shape (num_params, num_plans)
          where near-optimal plans have label 1 and rest are 0.
    """
    # Apply preprocessing to parameter columns.
    self.apply_preprocessing(query_execution_df)

    # Get the list of column names and cast any required columns to the
    # appropriate type.
    column_names = self.get_parameter_column_names()
    query_execution_df = trainer_util.cast_df_columns(query_execution_df,
                                                      self._predicate_metadata)
    # Default plans may be duplicated.
    query_execution_df.drop_duplicates(column_names + ["plan_id"], inplace=True)

    # If default relative, extract default latencies and then prune
    # query_execution_df to only contain plans in the plan cover. Default plans
    # not in the plan cover will be removed.
    if default_relative:
      default_latencies = query_execution_df[
          query_execution_df["is_default"]]["latency_ms"]
      query_execution_df = query_execution_df[
          query_execution_df["plan_id"].isin(self._plan_id_to_index)]

    grouped = query_execution_df.groupby(column_names, sort=False)
    plan_latencies = np.array(list(grouped["latency_ms"].apply(np.array)))

    # Get target outputs.
    target = np.zeros(plan_latencies.shape)
    optimal_latencies = np.min(plan_latencies, axis=1)
    if default_relative:
      thresholds = default_latencies - (
          (default_latencies - optimal_latencies) / near_optimal_threshold)
    else:
      thresholds = optimal_latencies * near_optimal_threshold
    target[plan_latencies <= thresholds[:, None]] = 1

    # Split the columns since they may be different types and need to be passed
    # into the model as multiple inputs.
    param_values = grouped.sum().index.to_frame()
    feature_values = [
        np.array(param_values.pop(n)) for n in param_values.columns
    ]
    return feature_values, target

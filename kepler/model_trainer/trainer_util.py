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

"""Helper functions Kepler modeling.
"""
import collections
import itertools
from typing import Any, Dict, List

import dateutil.parser
import numpy as np
import pandas as pd
import tensorflow as tf

from kepler.data_management import workload

JSON = Any


def get_vocabulary_by_max_marginal_improvement(query_execution_df: pd.DataFrame,
                                               predicate_metadata: JSON,
                                               column_index: int) -> List[str]:
  """Max marginal improvement vocabulary selection.

  Args:
    query_execution_df: Dataframe containing training data (e.g. from
      DatabaseClient.execute_timed_batch). This df must contain the default plan
      for each query instance.
    predicate_metadata: List containing metadata for each predicate.
    column_index: Index of relevant parameter.

  Returns:
    List of vocabulary values, sorted in descending order of their total
      potential improvement over the workload.
  """
  param_columns = get_parameter_column_names(len(predicate_metadata))
  vocabulary_to_improvement = collections.defaultdict(int)
  for params, params_df in query_execution_df.groupby(
      param_columns, sort=False):
    word = params[column_index]
    default_latency = np.min(params_df[params_df.is_default].latency_ms)
    optimal_latency = np.min(params_df.latency_ms)
    vocabulary_to_improvement[word] += default_latency - optimal_latency
  return [
      k for k, _ in sorted(
          vocabulary_to_improvement.items(), key=lambda x: x[1], reverse=True)
  ]


def construct_multihead_model_inputs(
    workload_params: workload.Workload,
    predicate_metadata: List[Any]) -> List[np.ndarray]:
  """Converts workload to input features for a multihead model.

  Args:
    workload_params: Workload of parameters.
    predicate_metadata: Predicate metadata used to determine preprocessing
      type.

  Returns:
    List of parameters by column.
  """
  if not workload_params.query_log:
    return []
  model_inputs = []
  for i in range(len(workload_params.query_log[0].parameters)):
    model_inputs.append(np.array([query_instance.parameters[i]
                                  for query_instance
                                  in workload_params.query_log]))
  _apply_preprocessing(model_inputs, predicate_metadata)
  return model_inputs


def _apply_preprocessing(model_inputs: List[np.ndarray],
                         predicate_metadata: List[Any]) -> None:
  """Applies preprocessing in-place to feature inputs.

  Args:
    model_inputs: List of feature arrays corresponding to parameters in order.
    predicate_metadata: Feature metadata for each predicate.
  """
  for i, predicate in enumerate(predicate_metadata):
    transform = lambda x: x
    if predicate.get("preprocess_type") == "to_timestamp":
      transform = lambda x: dateutil.parser.parse(x).timestamp()
    model_inputs[i] = np.array(list(map(transform, model_inputs[i])))


def extract_params_subset_data(query_execution_df: pd.DataFrame,
                               subset_indices: np.ndarray,
                               param_column_names: List[str]) -> pd.DataFrame:
  """Extracts query execution data corresponding to a specified set of params.

  Args:
    query_execution_df: Query execution data formatted as in
      DatabaseClient.execute_timed_batch: one row per plan, params; ordered by
      params.
    subset_indices: Indices indicating which params to keep. These indices
      are on the unique set of params, in the order in which they appear.
    param_column_names: Names of param columns in query_execution_df.

  Returns:
    Dataframe containing only rows corresponding to params specified by
      subset_indices.
  """
  params_groupby = query_execution_df.groupby(param_column_names)
  param_indices = params_groupby.indices
  all_params = sorted(list(param_indices.keys()),
                      key=lambda x: np.min(param_indices[x]))
  all_indices = [param_indices[all_params[i]] for i in subset_indices]
  all_indices = list(itertools.chain.from_iterable(all_indices))
  return query_execution_df.iloc[all_indices]


def cast_df_columns(df: pd.DataFrame,
                    predicate_metadata: List[Dict[str, Any]]) -> pd.DataFrame:
  """Cast df columns to their appropriate types.

  Args:
    df: Dataframe containing columns of the form "param{i}" corresponding
      to the parameters (e.g. from DatabaseClient.execute_timed_batch).
    predicate_metadata: Metadata for each predicate in the query template.

  Returns:
    Copy of df with columns cast to appropriate types for training.
  """
  col_types = {
      f"param{i}": get_pd_type(p["data_type"])
      for i, p in enumerate(predicate_metadata)
  }
  return df.astype(col_types)


def get_pd_type(data_type: str) -> str:
  """Map a parameter type string to a pandas type."""
  if data_type == "int":
    return "int64"
  elif data_type == "float":
    return "float64"
  elif data_type == "text":
    return "string"

  raise ValueError(f"Unsupported parameter type: {data_type}")


def get_tf_type(data_type: str) -> tf.dtypes.DType:
  """Map a parameter type string to a TensorFlow type."""
  if data_type == "int":
    return tf.int64
  elif data_type == "float":
    return tf.float32
  elif data_type == "text":
    return tf.string

  raise ValueError(f"Unsupported parameter type: {data_type}")


def get_np_type(data_type: str) -> np.dtype:
  """Map a parameter type string to a numpy type."""
  if data_type == "int":
    return np.int64
  elif data_type == "float":
    return np.float64
  elif data_type == "text":
    return np.dtype("O")

  raise ValueError(f"Unsupported parameter type: {data_type}")


def get_parameter_column_names(parameter_count: int) -> List[str]:
  """Get column names for each parameter.

  Args:
    parameter_count: The number of parameter bindings in the query template.

  Returns:
    List containing all parameter names in order.
  """
  return [get_parameter_column_name(i) for i in range(parameter_count)]


def get_parameter_column_name(parameter_index: int) -> str:
  return f"param{parameter_index}"

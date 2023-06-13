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

"""Utilities to prepare and manipulate graphs."""

import collections
import dataclasses
import functools
import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

from kepler.data_management import database_simulator
from kepler.data_management import workload
from kepler.training_data_collection_pipeline import query_text_utils
from kepler.training_data_collection_pipeline import query_utils

JSON = Any

_UNDEFINED = 'Undefined'

# The database is expensive to load. By making it global, we pay the loading
# cost exactly once when using a multiprocessing pool to parallelize
# create_examples. It will get copied into each process.
distributed_database = None

# The distributed_query_manager is initialized per-process to open a database
# connection exactly once per process when using a multiprocessing pool to
# parallelize create_examples.
distributed_query_manager = None


def _init_per_process_global_query_manager(
    database_configuration: query_utils.DatabaseConfiguration):
  global distributed_query_manager
  distributed_query_manager = query_utils.QueryManager(database_configuration)


@dataclasses.dataclass
class NodeMetadata:
  """Holder for feature data collected while traversing a query plan tree."""
  node_type: str
  relation_name: str
  alias: str
  join_type: str
  parent_relationship: str
  startup_cost: float
  total_cost: float
  cardinality_estimate: int
  tree_depth_from_root: int
  tree_height_from_deepest_leaf: int
  subtree_size: int


def _extract_graph_components_helper(
    query_plan_node: JSON, parent_node_index: Optional[int],
    next_node_index: int, tree_depth: int, node_metadata: List[NodeMetadata],
    edge_sources: List[int], edge_targets: List[int]) -> Tuple[int, int]:
  """Manages recursive walk to extract features from query EXPLAIN plan."""

  current_node_index = next_node_index

  node_metadata.append(
      NodeMetadata(
          node_type=query_plan_node['Node Type'],
          relation_name=query_plan_node.get('Relation Name', _UNDEFINED),
          alias=query_plan_node.get('Alias', _UNDEFINED),
          join_type=query_plan_node.get('Join Type', _UNDEFINED),
          parent_relationship=query_plan_node.get('Parent Relationship',
                                                  _UNDEFINED),
          startup_cost=query_plan_node['Startup Cost'],
          total_cost=query_plan_node['Total Cost'],
          cardinality_estimate=query_plan_node['Plan Rows'],
          tree_depth_from_root=tree_depth,
          tree_height_from_deepest_leaf=0,
          subtree_size=0,
      ))
  assert current_node_index == (
      len(node_metadata) - 1), f'{current_node_index} vs {len(node_metadata)}'

  if parent_node_index is not None:
    edge_sources.append(current_node_index)
    edge_targets.append(parent_node_index)

  tree_height = 0
  subtree_size = 1
  if 'Plans' in query_plan_node:
    for child in query_plan_node['Plans']:
      child_tree_height, child_subtree_size = _extract_graph_components_helper(
          query_plan_node=child,
          parent_node_index=current_node_index,
          next_node_index=next_node_index + subtree_size,
          tree_depth=tree_depth + 1,
          node_metadata=node_metadata,
          edge_sources=edge_sources,
          edge_targets=edge_targets)

      tree_height = max(tree_height, child_tree_height)
      subtree_size += child_subtree_size

  # Update the parent node based on the information gathered from the subtree.
  node_metadata[current_node_index].tree_height_from_deepest_leaf = tree_height
  node_metadata[current_node_index].subtree_size = subtree_size

  return tree_height + 1, subtree_size


def _extract_graph_components(
    query_plan: JSON) -> Tuple[int, List[NodeMetadata], List[int], List[int]]:
  """Extracts graph structures and node features from query EXPLAIN plan.

  Please update print_to_dot() with any added features.

  Args:
    query_plan: Dict containing the explain plan (contents of top level key
      'Plan').

  Returns:
    Extracted components from the query_plan that will be used to construct a
    corresponding graph tensor:
      node_count: The number of nodes in the query plan.
      node_metadata: A list of NodeMetadata instances describing each node.
      edge_sources: A list of node indices describing the source of each graph
        edge.
      edge_targets: A list of node indices describing the target of each graph
        edge using the same ordering as edge_sources.
  """

  node_metadata = []
  edge_sources = []
  edge_targets = []

  _, child_subtree_size = _extract_graph_components_helper(
      query_plan_node=query_plan,
      parent_node_index=None,
      next_node_index=0,
      tree_depth=0,
      node_metadata=node_metadata,
      edge_sources=edge_sources,
      edge_targets=edge_targets)

  return child_subtree_size, node_metadata, edge_sources, edge_targets


def _compose_node_features(
    node_metadata: List[NodeMetadata]
) -> Dict[str, List[Optional[Union[str, float, int]]]]:
  features_map = collections.defaultdict(list)
  for metadata in node_metadata:
    metadata_dict = dataclasses.asdict(metadata)
    for key, value in metadata_dict.items():
      features_map[key].append(value)

  return features_map


def build_graph_tensor(query_plan: Any,
                       labels: Dict[str, float]) -> tfgnn.GraphTensor:
  """Builds a graph tensor from a Postgres EXPLAIN plan.

  Building a graph tensor involves:
    1. Parsing out nodes and edges.
    2. Extracting features from the EXPLAIN plan and encoding them in the nodes.
    3. Populating the graph context with the label for training.

  Args:
    query_plan: The EXPLAIN plan from Postgres in JSON format.
    labels: The labels to set in the graph context.

  Returns:
    A graph tensor that represents the query_plan as a graph with node features.
  """
  node_count, node_metadata, edge_sources, edge_targets = _extract_graph_components(
      query_plan['Plan'])

  node_features_map = _compose_node_features(node_metadata)
  for feature_name, feature_values in node_features_map.items():
    node_features_map[feature_name] = tf.ragged.constant(feature_values)

  context_labels_map = {}
  for label, value in labels.items():
    context_labels_map[label] = tf.ragged.constant([float(value)])

  return tfgnn.GraphTensor.from_pieces(
      context=tfgnn.Context.from_fields(features=context_labels_map),
      node_sets={
          'node':
              tfgnn.NodeSet.from_fields(
                  sizes=tf.ragged.constant([node_count]),
                  features=node_features_map)
      },
      edge_sets={
          'edge':
              tfgnn.EdgeSet.from_fields(
                  sizes=tf.ragged.constant([len(edge_sources)]),
                  features={},
                  adjacency=tfgnn.Adjacency.from_indices(
                      source=('node', edge_sources),
                      target=('node', edge_targets)))
      })


def convert_to_example(
    graph_tensor: tfgnn.GraphTensor,
    graph_spec: Optional[tfgnn.GraphTensorSpec] = None) -> str:
  """Converts graph tensor to serialized tf.train.Example.

  If a graph_spec is provided, each serialized graph tensor is parsed again
  using the graph_spec to ensure it was constructed to spec.

  Args:
    graph_tensor: The graph tensor to convert to tf.train.Example.
    graph_spec: Optional spec provided to validate the structure of
      graph_tensor.

  Returns:
    The graph_tensor as a serialized tf.train.Example.
  """
  example = tfgnn.write_example(graph_tensor).SerializeToString()

  if graph_spec:
    tfgnn.parse_single_example(graph_spec, tf.constant(example))

  return example


def _construct_example(graph_spec: tfgnn.GraphTensorSpec, query_id: str,
                       plan_id: int, hinted_query: str,
                       query_instance: workload.QueryInstance) -> str:
  """Collects components to build graph tensor and convert to example."""
  query_plan = distributed_query_manager.get_query_plan(
      hinted_query, query_instance.parameters)
  total_cost = query_plan['Plan']['Total Cost']
  latency_ms, _ = distributed_database.execute_timed(
      database_simulator.PlannedQuery(
          query_id=query_id,
          plan_id=plan_id,
          parameters=query_instance.parameters))

  graph_tensor = build_graph_tensor(
      query_plan=query_plan,
      labels={
          'total_cost': total_cost,
          'latency_ms': latency_ms
      })
  return convert_to_example(graph_tensor=graph_tensor, graph_spec=graph_spec)


def create_examples(database_configuration: query_utils.DatabaseConfiguration,
                    graph_schema_filename: str,
                    query_id: str,
                    templates: Any,
                    query_execution_data: JSON,
                    query_execution_metadata: JSON,
                    plan_hints: JSON,
                    write_examples_fn: Callable[[List[str], int], None],
                    limit: Optional[int] = None,
                    multiprocessing_chunksize: int = 100) -> None:
  """Generates graph tensors per parameters and plan combination.

  The parameters follow a workload.all() generated from query_execution_data.

  The plans are selected based on the plan cover found in
  query_execution_metadata.

  Args:
    database_configuration: The configuration describing the database
      connection.
    graph_schema_filename: The path to the file containing the graph schema.
    query_id: The query id describing all the query plans.
    templates: A mapping from query id to the templatized SQL query text.
    query_execution_data: Execution data structure that defines all the known
      information regarding query plans, parameter bindings, and latencies. The
      format is a series of nested dicts, typically parsed from a JSON file.
    query_execution_metadata: Metadata from query execution that is expected to
      contain a "plan_cover" entry.
    plan_hints: A mapping from query id to all the pg_hint_plan hints generated
      for that query.
    write_examples_fn: A callback function called once per plan after all the
      examples for that plan are generated. The function is expected to accept
      the list of examples (serialized to string) and the plan id.
    limit: The number of parameter values to build graph tensors for.
    multiprocessing_chunksize: The chunksize used when splitting work in imap.
      This is primarily exposed to ensure tests can run with multiple processes
      by setting a low multiprocessing_chunksize.
  """

  global distributed_database
  distributed_database = database_simulator.DatabaseSimulator(
      query_execution_data=query_execution_data,
      query_execution_metadata=query_execution_metadata,
      estimator=database_simulator.LatencyEstimator.MIN)

  graph_spec = tfgnn.create_graph_spec_from_schema_pb(
      tfgnn.read_schema(graph_schema_filename))

  plan_ids = workload.KeplerPlanDiscoverer(
      query_execution_metadata=query_execution_metadata).plan_ids
  workload_all = workload.WorkloadGenerator(
      query_execution_data=query_execution_data).all()

  for plan_id in plan_ids:
    hinted_query = query_text_utils.get_hinted_query(
        query=templates[query_id]['query'],
        hints=plan_hints[query_id][plan_id]['hints'])
    with multiprocessing.Pool(
        initializer=_init_per_process_global_query_manager,
        initargs=[database_configuration]) as pool:
      examples = pool.map(
          func=functools.partial(_construct_example, graph_spec, query_id,
                                 plan_id, hinted_query),
          iterable=workload_all.query_log[:limit],
          chunksize=multiprocessing_chunksize)

    write_examples_fn(examples, plan_id)


def _construct_dot_node_label_from_features(node_feature_map,
                                            index: int) -> str:
  """Constructs the label for a node in dot graph based on feature values."""
  tokens = []
  for key, value in node_feature_map.items():
    feature_value = value[index].numpy()
    if isinstance(feature_value, bytes):
      tokens.append(f'{key}: {feature_value.decode("utf-8")}')
    elif isinstance(feature_value, np.float32):
      tokens.append(f'{key}: {feature_value:.2f}')
    else:
      tokens.append(f'{key}: {feature_value}')
  return '\n'.join(tokens)


# TODO(lyric): Implement a flag-controlled verbose mode for all features.
def to_dot(graph_tensor: tfgnn.GraphTensor) -> str:
  """Builds a string representing the graph_tensor in dot format.

  Args:
    graph_tensor: The graph tensor to encode in dot.

  Returns:
    The graph_tensor's structure and features encoded in dot.
  """
  graph_labels = [
      f'{label}: {value.numpy()[0]:.2f}'
      for label, value in graph_tensor.context.get_features_dict().items()
  ]
  graph_label = '\n'.join(['graph-level'] + graph_labels)

  node_sets_features_dict = graph_tensor.node_sets['node'].get_features_dict()
  features_count = len(node_sets_features_dict[next(
      iter(node_sets_features_dict))])
  nodes = '\n'.join([
      f'{i} [label="{_construct_dot_node_label_from_features(node_sets_features_dict, i)}"]'
      for i in range(features_count)
  ])

  edges = '\n'.join([
      f'{source} -> {target}'
      for source, target in zip(graph_tensor.edge_sets['edge'].adjacency.source,
                                graph_tensor.edge_sets['edge'].adjacency.target)
  ])

  return f'digraph D {{\nlabel="{graph_label}"\n{nodes}\n{edges} }}'

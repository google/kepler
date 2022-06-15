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

"""Tests for graph_tensor_builder.py."""

import json
import os
from typing import Any, List

import tensorflow as tf
import tensorflow_gnn as tfgnn

from kepler.training_data_collection_pipeline import graph_tensor_builder
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util
from google3.pyglib import resources
from absl.testing import absltest
from absl.testing import parameterized

# Typing aliases.
JSON = Any

_GRAPH_SCHEMA_FILE = "kepler/training_data_collection_pipeline/query_plan_graph_schema.pbtxt"
_TEST_DATA_DIR = "kepler/training_data_collection_pipeline/testdata"


def _graph_key_invariants_check(query_plan: Any):
  """Ensures every node has certain invariants."""

  assert "Plan Rows" in query_plan
  if "Plans" in query_plan:
    for child in query_plan["Plans"]:
      _graph_key_invariants_check(child)


class GraphTensorBuilderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    plan_path = os.path.join(_TEST_DATA_DIR,
                             "generate_candidates_explain_plans.json")
    explain_data = json.loads(resources.GetResource(plan_path))
    self.explain_plan = explain_data["output"]["q0_0"][0]["result"]
    _graph_key_invariants_check(self.explain_plan["Plan"])

  def test_to_dot(self):
    """Verifies to_dot encodes the graph properties."""

    node_count = 4
    node_types = ["A", "B", "C", "B"]
    cardinality_estimates = [1, 2, 3, 4]
    edge_sources = [1, 2, 3]
    edge_targets = [0, 0, 2]

    graph_tensor = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={"total_cost": tf.ragged.constant([42])}),
        node_sets={
            "node":
                tfgnn.NodeSet.from_fields(
                    sizes=tf.ragged.constant([node_count]),
                    features={
                        "node_type":
                            tf.ragged.constant(node_types),
                        "cardinality_estimate":
                            tf.ragged.constant(cardinality_estimates),
                    })
        },
        edge_sets={
            "edge":
                tfgnn.EdgeSet.from_fields(
                    sizes=tf.ragged.constant([len(edge_sources)]),
                    features={},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("node", edge_sources),
                        target=("node", edge_targets)))
        })

    # Copy and pasted directly from print() output to make test expected output
    # updates straightforward.
    expected = """digraph D {
label="graph-level
total_cost: 42.00"
0 [label="node_type: A
cardinality_estimate: 1"]
1 [label="node_type: B
cardinality_estimate: 2"]
2 [label="node_type: C
cardinality_estimate: 3"]
3 [label="node_type: B
cardinality_estimate: 4"]
1 -> 0
2 -> 0
3 -> 2 }"""
    self.assertEqual(graph_tensor_builder.to_dot(graph_tensor), expected)

  def test_build_graph_tensor_basic(self):
    """Verifies construction of a single graph tensor from an EXPLAIN plan.

    The structure of the graph is verified by to_dot().
    """

    graph_tensor = graph_tensor_builder.build_graph_tensor(
        self.explain_plan, {
            "total_cost": 42,
            "latency_ms": 4.2
        })
    expected = """digraph D {
label="graph-level
total_cost: 42.00
latency_ms: 4.20"
0 [label="node_type: Hash Join
relation_name: Undefined
alias: Undefined
join_type: Inner
parent_relationship: Undefined
startup_cost: 21.71
total_cost: 62.90
cardinality_estimate: 11
tree_depth_from_root: 0
tree_height_from_deepest_leaf: 2
subtree_size: 4"]
1 [label="node_type: Seq Scan
relation_name: foo
alias: foo
join_type: Undefined
parent_relationship: Outer
startup_cost: 0.00
total_cost: 32.60
cardinality_estimate: 2260
tree_depth_from_root: 1
tree_height_from_deepest_leaf: 0
subtree_size: 1"]
2 [label="node_type: Hash
relation_name: Undefined
alias: Undefined
join_type: Undefined
parent_relationship: Inner
startup_cost: 21.70
total_cost: 21.70
cardinality_estimate: 1
tree_depth_from_root: 1
tree_height_from_deepest_leaf: 1
subtree_size: 2"]
3 [label="node_type: Seq Scan
relation_name: bar
alias: bar
join_type: Undefined
parent_relationship: Outer
startup_cost: 0.00
total_cost: 21.70
cardinality_estimate: 1
tree_depth_from_root: 2
tree_height_from_deepest_leaf: 0
subtree_size: 1"]
1 -> 0
2 -> 0
3 -> 2 }"""
    self.assertEqual(graph_tensor_builder.to_dot(graph_tensor), expected)

  def test_convert_to_example(self):
    """Verifies schema enforcement."""

    graph_tensor = graph_tensor_builder.build_graph_tensor(
        self.explain_plan, {
            "total_cost": 42,
            "latency_ms": 4.2
        })
    example = graph_tensor_builder.convert_to_example(graph_tensor)
    self.assertNotEmpty(example)

    graph_schema = tfgnn.read_schema(
        resources.GetResourceFilename(_GRAPH_SCHEMA_FILE))
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    example = graph_tensor_builder.convert_to_example(graph_tensor, graph_spec)
    self.assertNotEmpty(example)

    bad_graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "node":
                tfgnn.NodeSet.from_fields(
                    sizes=tf.ragged.constant([3]),
                    features={"node_type": tf.ragged.constant([1, 2, 3])})
        })
    self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                           "Expected type: string, Actual type: int64",
                           graph_tensor_builder.convert_to_example,
                           bad_graph_tensor, graph_spec)


_HINTS_UNUSED = """{"hints": "/*+ HashJoin(bar foo) Leading((bar foo)) */", "source": "default"}"""
_HINTS = """[{hints_0}, {hints_1}, {hints_2}]""".format(
    hints_0=test_util.HINTS_0, hints_1=_HINTS_UNUSED, hints_2=test_util.HINTS_1)
_QUERY_EXECUTION_METADATA = {test_util.TEST_QUERY_ID: {"plan_cover": [0, 2]}}
_QUERY_EXECUTION_DATA = {
    test_util.TEST_QUERY_ID: {
        "0####alpha": {
            "default":
                0,
            "results": [[{
                "duration_ms": 1
            }], [{
                "duration_ms": 2
            }], [{
                "duration_ms": 3
            }]]
        },
        "1####bravo": {
            "default":
                0,
            "results": [[{
                "duration_ms": 11
            }], [{
                "duration_ms": 12
            }], [{
                "duration_ms": 13
            }]]
        },
        "1####charlie": {
            "default":
                0,
            "results": [[{
                "duration_ms": 21
            }], [{
                "duration_ms": 22
            }], [{
                "duration_ms": 23
            }]]
        }
    }
}


class CreateExamplesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self.database_configuration = query_utils.DatabaseConfiguration(
        dbname=self._test_database.dbname,
        user=test_util.USER,
        password=test_util.PASSWORD)
    self._query_manager = query_utils.QueryManager(self.database_configuration)
    test_util.populate_database(self._query_manager)

    self.templates = {
        test_util.TEST_QUERY_ID: json.loads(test_util.TEST_TEMPLATE_STRING)
    }
    self.hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS)}

    self.graph_schema_filename = resources.GetResourceFilename(
        _GRAPH_SCHEMA_FILE)
    self.graph_spec = tfgnn.create_graph_spec_from_schema_pb(
        tfgnn.read_schema(self.graph_schema_filename))

  def tearDown(self):
    self._test_database.drop()
    super().tearDown()

  @parameterized.named_parameters(
      dict(testcase_name="none", limit=0), dict(testcase_name="1", limit=1),
      dict(testcase_name="all", limit=3),
      dict(testcase_name="no limit", limit=None))
  def test_create_examples(self, limit: int):

    examples_map = {}

    def write_examples(examples: List[str], plan_id: int) -> None:
      nonlocal examples_map
      self.assertNotIn(plan_id, examples_map)
      examples_map[plan_id] = examples

    graph_tensor_builder.create_examples(
        database_configuration=self.database_configuration,
        graph_schema_filename=self.graph_schema_filename,
        query_id=test_util.TEST_QUERY_ID,
        templates=self.templates,
        query_execution_data=_QUERY_EXECUTION_DATA,
        query_execution_metadata=_QUERY_EXECUTION_METADATA,
        plan_hints=self.hints,
        write_examples_fn=write_examples,
        limit=limit,
        multiprocessing_chunksize=1)

    expected_map = {
        0: {
            "total_costs": [190, 190, 190],
            "latencies": [1, 11, 21],
            "node_counts": [6] * 3,
            "edge_counts": [5] * 3
        },
        2: {
            "total_costs": [62, 62, 62],
            "latencies": [3, 13, 23],
            "node_counts": [4] * 3,
            "edge_counts": [3] * 3
        }
    }

    self.assertEqual(examples_map.keys(), expected_map.keys())
    for plan_id in examples_map:
      graph_tensors = [
          tfgnn.parse_single_example(self.graph_spec, example)
          for example in examples_map[plan_id]
      ]

      total_costs = [
          int(graph_tensor.context.features["total_cost"][0].numpy())
          for graph_tensor in graph_tensors
      ]
      self.assertEqual(total_costs,
                       expected_map[plan_id]["total_costs"][:limit])

      latencies = [
          int(graph_tensor.context.features["latency_ms"][0].numpy())
          for graph_tensor in graph_tensors
      ]
      self.assertEqual(latencies, expected_map[plan_id]["latencies"][:limit])

      node_counts = [
          len(graph_tensor.node_sets["node"].features["node_type"])
          for graph_tensor in graph_tensors
      ]
      self.assertEqual(node_counts,
                       expected_map[plan_id]["node_counts"][:limit])

      edge_counts = [
          len(graph_tensor.edge_sets["edge"].adjacency.source)
          for graph_tensor in graph_tensors
      ]
      self.assertEqual(edge_counts,
                       expected_map[plan_id]["edge_counts"][:limit])


if __name__ == "__main__":
  absltest.main()

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

"""Query parsing utilties to extract attributes for the prediction server."""

import re
from typing import Any, Dict, List, Optional

import pglast

_COMMENT_START = "/*+"
_COMMENT_END = "*/"

_SKIP_ATTRIBUTES = ["location"]

_PARAM_REGEX = r"@param[0-9]*"
_PARAM_SINGLE_QUOTE_REGEX = r"'@param[0-9]*'"
_PARAM_DOUBLE_QUOTE_REGEX = r"\"@param[0-9]*\""
_PARAM_TOKEN_CHECKER = re.compile(_PARAM_REGEX)


def _add_quotes_to_param(match):
  return f"'{match.group()}'"


def _strip_quotes_from_param(match):
  return match.group()[1:-1]


def _canonize_params(query: str) -> str:
  """Ensures all params are single-quoted."""
  query = re.sub(_PARAM_SINGLE_QUOTE_REGEX, _strip_quotes_from_param, query)
  query = re.sub(_PARAM_DOUBLE_QUOTE_REGEX, _strip_quotes_from_param, query)
  return re.sub(_PARAM_REGEX, _add_quotes_to_param, query)


def _get_nodes_flattened(query: str) -> List[Any]:
  """Flattens out the query nodes and values from the parsed AST."""
  root = pglast.Node(pglast.parse_sql(query))
  nodes_flattened = []
  _get_nodes_flattened_helper(root, nodes_flattened)
  return nodes_flattened


def _get_nodes_flattened_helper(node, nodes_flattened: List[Any]):
  """Executes a step in the recursive tree walk to flatten the parsed AST."""
  if isinstance(node, pglast.node.Scalar):
    if node.parent_attribute in _SKIP_ATTRIBUTES:
      return

    nodes_flattened.append(node.value)
    return
  elif isinstance(node, pglast.node.Node):
    nodes_flattened.append(node.node_tag)
  for child in node:
    _get_nodes_flattened_helper(child, nodes_flattened)


def _get_param_indices_map(nodes_flattened: List[Any]) -> Dict[str, int]:
  """Maps params to positions to extract param values from query instances."""
  param_indices_map = {}
  for i, token in enumerate(nodes_flattened):
    if _PARAM_TOKEN_CHECKER.search(str(token)):
      param_indices_map[token] = i

  return param_indices_map


class ParamExtractor:
  """Extracts param values from query instances of a given query template."""

  def __init__(self, query_template: str):
    canonized_query_template = _canonize_params(query_template)
    nodes_flattened = _get_nodes_flattened(canonized_query_template)
    self._node_count = len(nodes_flattened)
    self._param_indices_map = _get_param_indices_map(nodes_flattened)

    # Check for parameter number issues.
    self._extract_params(nodes_flattened)

  def _extract_params(self, nodes_flattened: List[Any]) -> List[Any]:
    params = []
    for i in range(len(self._param_indices_map)):
      param_key = f"@param{i}"

      if param_key not in self._param_indices_map:
        raise ValueError("Param indices are not consecutive starting with 0.")

      params.append(nodes_flattened[self._param_indices_map[param_key]])
    return params

  def get_params(self, query_instance: str) -> List[Any]:
    """Extracts parameter binding values from a query instance.

    The caller is expected to cast data types as required.

    The current implementation presumes the limitations used in the current
    Kepler iteration. For example, IN lists are expected to only contain a
    single element.

    Args:
      query_instance: An instance with parameter values substituted into the
        query template used to instantiate this instance of ParamExtractor.

    Returns:
      A list of param binding values found in the query instance. The order of
      the values in the list matches the @param tokens from the
      query template in sorted order, not appearance order.
    """
    nodes_flattened = _get_nodes_flattened(query_instance)
    if len(nodes_flattened) != self._node_count:
      raise ValueError(
          f"Mismatch in flattened query tree size between query instance ({len(nodes_flattened)}) and template ({self._node_count})."
      )

    return self._extract_params(nodes_flattened)


def extract_comment_content(query: str) -> Optional[str]:
  """Extracts the content from the query comment syntax.

  This function assumes a protocol where the comment is of the following exact
  format:
  /*+ <comment content> */ SELECT ...

  Args:
    query: The SQL query text to parse.

  Returns:
    The comment content if found, None otherwise.
  """
  try:
    return query[len(_COMMENT_START):query.index(_COMMENT_END)].strip()
  except ValueError:
    return None

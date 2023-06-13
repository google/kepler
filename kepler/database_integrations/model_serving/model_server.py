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

"""Server returning predicted query plan hints for provided query instances."""

import socket

from typing import Any, Dict

from kepler.database_integrations.model_serving import query_parsing_utils
from kepler.model_trainer import model_base

_READ_BYTES_COUNT = 2048


class ModelServer:
  """Serves query plan hints for provided query instances.

  The server runs model inference for known query ids and returns a plan
  prediction if it is able to make one. The plan prediction is conveyed via a
  hint string.
  """

  def __init__(
      self,
      host: str,
      port: int,
      param_extractors: Dict[str, query_parsing_utils.ParamExtractor],
      plan_hints: Any,
      predictors: Dict[str, model_base.ModelPredictorBase],
      read_bytes_count: int = _READ_BYTES_COUNT,
  ):
    """Prepare a server to provide hints for queries.

    Args:
      host: The host address for the model server.
      port: The port for the model server to use.
      param_extractors: A mapping from query id to its corresponding
        ParamExtractor.
      plan_hints: A mapping from query id to pg_hint_plan hints representing the
        set of query plans for execution.
      predictors: A mapping from query id to its corresponding query plan
        predictor.
      read_bytes_count: The number of bytes to read at a time from the socket.
        This is exposed primarily for testing.

    Raises:
      ValueError: If predictors contains query_ids not found in plan_hints or
        param_extractors.
    """
    self._host = host
    self._port = port
    self._param_extractors = param_extractors
    self._plan_hints = plan_hints
    self._predictors = predictors
    self._read_bytes_count = read_bytes_count

    extra_query_ids = self._predictors.keys() - self._param_extractors.keys()
    if extra_query_ids:
      raise ValueError(
          "Provided predictors contains the following query ids not found in"
          f" param_extractors. {extra_query_ids}"
      )

    extra_query_ids = self._predictors.keys() - self._plan_hints.keys()
    if extra_query_ids:
      raise ValueError(
          "Provided predictors contains the following query ids not found in"
          f" plan_hints. {extra_query_ids}"
      )

  def run(self) -> None:
    """Run the main server loop to await connections and handle requests."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.bind((self._host, self._port))
      s.listen()

      while True:
        conn, _ = s.accept()
        with conn:
          while True:
            data = conn.recv(4)
            data_length = int.from_bytes(data, "little")
            data = b""
            while data_length > len(data):
              data += conn.recv(self._read_bytes_count)

            if not data:
              break

            query = data.decode()
            query_id = query_parsing_utils.extract_comment_content(query)

            if query_id not in self._predictors:
              response = " "
            else:
              query_params = self._param_extractors[query_id].get_params(query)
              plan_ids, _ = self._predictors[query_id].predict(query_params)
              assert len(plan_ids) == 1
              plan_id = plan_ids[0]

              if plan_id is not None:
                response = query_parsing_utils.extract_comment_content(
                    self._plan_hints[query_id][int(plan_id)]["hints"]
                )
              else:
                response = " "

            conn.sendall(response.encode())

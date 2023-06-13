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

"""Tests for model_server."""

import json
import multiprocessing
import socket
import time

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kepler.database_integrations.model_serving import model_server
from kepler.database_integrations.model_serving import query_parsing_utils
from kepler.model_trainer import model_base
from kepler.training_data_collection_pipeline import test_util
from kepler.training_data_collection_pipeline import utils
from absl.testing import absltest
from absl.testing import parameterized

_LOCAL_HOST = "127.0.0.1"
_KEPLER_PORT = 30709


_QUERY_REQUEST = "/*+ {query_id} */ {query}"

_HINTS_UNUSED = """{"hints": "/*+ unused */", "source": "unused"}"""
_HINTS = """[{hints_0}, {hints_unused}, {hints_1}]""".format(
    hints_0=test_util.HINTS_0,
    hints_unused=_HINTS_UNUSED,
    hints_1=test_util.HINTS_1,
)

_PARAMS_HINT_0 = [0, "hi"]
_PARAMS_HINT_1 = [2, "hi"]
_PARAMS_NO_HINT = [0, "skip"]


def _send_request(query_id: str, query: str) -> str:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((_LOCAL_HOST, _KEPLER_PORT))
    request = _QUERY_REQUEST.format(query_id=query_id, query=query).encode()
    s.sendall(len(request).to_bytes(4, "little"))
    s.sendall(request)
    return s.recv(2048).decode()


class ConcreteModelPredictor(model_base.ModelPredictorBase):
  """Make non-abstract class for testing."""

  def predict(
      self, params: List[Any]
  ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if params[1] == "skip":
      return np.array([None]), None

    return np.array([params[0]]), None


class ModelServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._query_template = json.loads(test_util.TEST_TEMPLATE_STRING)["query"]
    self._param_extractors = {
        test_util.TEST_QUERY_ID: query_parsing_utils.ParamExtractor(
            query_template=self._query_template
        )
    }
    self._plan_hints = {test_util.TEST_QUERY_ID: json.loads(_HINTS)}
    self._predictors = {test_util.TEST_QUERY_ID: ConcreteModelPredictor()}
    self._server_process = None

  def tearDown(self):
    if self._server_process:
      self._server_process.terminate()
      while self._server_process.is_alive():
        time.sleep(0.01)

    super().tearDown()

  def _start_server(
      self,
      read_bytes_count: int = 1024,
      param_extractors=None,
      plan_hints=None,
  ) -> None:
    param_extractors = (
        param_extractors
        if param_extractors is not None
        else self._param_extractors
    )
    plan_hints = plan_hints if plan_hints is not None else self._plan_hints

    server = model_server.ModelServer(
        host=_LOCAL_HOST,
        port=_KEPLER_PORT,
        param_extractors=param_extractors,
        plan_hints=plan_hints,
        predictors=self._predictors,
        read_bytes_count=read_bytes_count,
    )
    self._server_process = multiprocessing.Process(target=server.run)
    self._server_process.start()

    # Block until the server is ready to accept connections.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      while True:
        try:
          s.connect((_LOCAL_HOST, _KEPLER_PORT))
          return
        except ConnectionRefusedError:
          time.sleep(0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name="read_bytes_count larger than query",
          read_bytes_count=1024,
      ),
      dict(
          testcase_name="read_bytes_count smaller than query",
          read_bytes_count=5,
      ),
  )
  def test_read_bytes_count(self, read_bytes_count: int):
    self._start_server(read_bytes_count=read_bytes_count)

    hints = _send_request(
        query_id=test_util.TEST_QUERY_ID,
        query=utils.substitute_query_params(
            query=self._query_template, params=_PARAMS_HINT_0
        ),
    )
    self.assertIn("MergeJoin", hints)

  @parameterized.named_parameters(
      dict(
          testcase_name="hint 0",
          params=_PARAMS_HINT_0,
          expected="MergeJoin(foo bar) Leading((foo bar))",
      ),
      dict(
          testcase_name="hint 1",
          params=_PARAMS_HINT_1,
          expected="HashJoin(foo bar) Leading((foo bar))",
      ),
      dict(testcase_name="no hints", params=_PARAMS_NO_HINT, expected=" "),
  )
  def test_get_hints(self, params: List[Any], expected: str):
    self._start_server()

    hints = _send_request(
        query_id=test_util.TEST_QUERY_ID,
        query=utils.substitute_query_params(
            query=self._query_template, params=params
        ),
    )
    self.assertEqual(hints, expected)

  @parameterized.named_parameters(
      dict(testcase_name="unknown id", query_id="q22"),
      dict(testcase_name="empty", query_id=""),
  )
  def test_query_id_without_predictor(self, query_id: str):
    self._start_server()

    hints = _send_request(
        query_id=query_id,
        query=utils.substitute_query_params(
            query=self._query_template, params=_PARAMS_HINT_1
        ),
    )
    self.assertEqual(hints, " ")

  def test_inconsistent_init(self):
    plan_hints = {"q22": json.loads(_HINTS)}
    self.assertRaisesRegex(
        ValueError,
        "not found in plan_hints",
        self._start_server,
        plan_hints=plan_hints,
    )

    param_extractors = {
        "q23": query_parsing_utils.ParamExtractor(
            query_template=self._query_template
        )
    }
    self.assertRaisesRegex(
        ValueError,
        "not found in param_extractors",
        self._start_server,
        param_extractors=param_extractors,
    )


if __name__ == "__main__":
  absltest.main()

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

"""Tests for pg_hint_plan extension."""

import multiprocessing
import socket
import time

import psycopg2

from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import test_util
from absl.testing import absltest
from absl.testing import parameterized

_LOCAL_HOST = "127.0.0.1"
_KEPLER_PORT = 30709

_SET_PORT = "SET pg_hint_plan.kepler_port TO {port};"
_SET_HOST = "SET pg_hint_plan.kepler_host TO '127.0.0.1';"
_RESET_HOST = "RESET pg_hint_plan.kepler_host;"
_QUERY = "SELECT COUNT(*) FROM foo JOIN bar ON x = a JOIN baz ON k = a;"
_DEFAULT_JOIN_ORDER = "Leading(((foo baz) bar))"
_FORCED_JOIN_ORDER = "Leading(((bar foo) baz))"

_TRIGGER_KEPLER = "hint"
_TRIGGER_READ_ERROR = "read error"


def _run_server(read_bytes_count: int) -> None:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((_LOCAL_HOST, _KEPLER_PORT))
    s.listen()

    b_trigger_kepler = _TRIGGER_KEPLER.encode("ascii")
    b_trigger_read_error = _TRIGGER_READ_ERROR.encode("ascii")
    while True:
      conn, _ = s.accept()
      with conn:
        while True:
          data = conn.recv(4)
          data_length = int.from_bytes(data, "little")
          data = b""
          while data_length > len(data):
            data += conn.recv(read_bytes_count)

          if not data:
            break

          if b_trigger_kepler in data:
            response = _FORCED_JOIN_ORDER
          elif b_trigger_read_error in data:
            return
          else:
            response = " "

          conn.sendall(response.encode())


def _trigger_kepler(query: str) -> str:
  return f"/*+ {_TRIGGER_KEPLER} */ {query}"


def _trigger_read_error(query: str) -> str:
  return f"/*+ {_TRIGGER_READ_ERROR} */ {query}"


class PgHintPlanExtensionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_database = test_util.TestDatabase()
    self._query_manager = query_utils.QueryManager(
        database_configuration=query_utils.DatabaseConfiguration(
            dbname=self._test_database.dbname,
            user=test_util.USER,
            password=test_util.PASSWORD))
    test_util.populate_database(self._query_manager)
    # Some tests read from pg_stats tables, which requires running ANALYZE.
    # Warning: Removing/adding this affects cardinality estimates.
    self._query_manager.run_analyze()

    self._query_manager.execute_and_commit(_SET_PORT.format(port=_KEPLER_PORT))
    self._server_process = None

  def tearDown(self):
    if self._server_process:
      self._server_process.terminate()
      while self._server_process.is_alive():
        time.sleep(0.01)

    self._test_database.drop()
    super().tearDown()

  def start_server(self, read_bytes_count: int) -> None:
    self._server_process = multiprocessing.Process(
        target=_run_server, args=(read_bytes_count,))
    self._server_process.start()
    # Block until the server is ready to accept connections.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      while True:
        try:
          s.connect((_LOCAL_HOST, _KEPLER_PORT))
          return
        except ConnectionRefusedError:
          time.sleep(.01)

  @parameterized.named_parameters(
      dict(
          testcase_name="read_bytes_count larger than query",
          read_bytes_count=1024),
      dict(
          testcase_name="read_bytes_count smaller than query",
          read_bytes_count=5),
  )
  def test_pg_hint_plan_extension(self, read_bytes_count: int):
    """Tests receiving hints while toggling the extension on and off."""
    self.start_server(read_bytes_count=read_bytes_count)
    self._query_manager.execute_and_commit(_SET_HOST)

    kepler_query = _trigger_kepler(_QUERY)
    # Repeat requests to ensure there's no lag in hinting due to state
    # mismanagement.
    for _ in range(3):
      for _ in range(3):
        hints = pg_plan_hint_extractor.get_single_query_hints_with_plan(
            query_manager=self._query_manager, query=_QUERY, params=None)[0]
        self.assertIn(_DEFAULT_JOIN_ORDER, hints)

      for _ in range(3):
        hints = pg_plan_hint_extractor.get_single_query_hints_with_plan(
            query_manager=self._query_manager, query=kepler_query,
            params=None)[0]
        self.assertIn(_FORCED_JOIN_ORDER, hints)

      self._query_manager.execute_and_commit(_RESET_HOST)
      for _ in range(3):
        hints = pg_plan_hint_extractor.get_single_query_hints_with_plan(
            query_manager=self._query_manager, query=kepler_query,
            params=None)[0]
        self.assertIn(_DEFAULT_JOIN_ORDER, hints)
      self._query_manager.execute_and_commit(_SET_HOST)

  def test_create_connection_error(self):
    with self.assertRaisesRegex(psycopg2.InternalError,
                                "Unable to create connection to Kepler server"):
      # The COMMIT triggers the Kepler connection creation attempt.
      self._query_manager.execute_and_commit(_SET_HOST)

    # TODO(lyric): Test that we can get out of this issue by RESET-ing
    # pg_hint_plan.kepler_host. This works at the SQL command prompt, but not in
    # the tests at present due to errors around failed transaction. Debugging
    # required.

  def test_unable_to_receive_hints(self):
    self.start_server(read_bytes_count=1024)
    self._query_manager.execute_and_commit(_SET_HOST)

    with self.assertRaisesRegex(psycopg2.InternalError,
                                "Unable to receive hint from Kepler server"):
      self._query_manager.execute(_trigger_read_error(_QUERY))


if __name__ == "__main__":
  absltest.main()

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

"""Provides simple generic utils."""

from typing import Any, List, Sequence


# TODO(b/199162711): Transition this script and downstream analysis scripts to a
# structured format instead of using _NAME_DELIMITER.
_NAME_DELIMITER = '####'


def get_params_as_string(params: List[Any]) -> str:
  return _NAME_DELIMITER.join([str(p) for p in params])


def substitute_query_params(query: str, params: Sequence[Any]) -> str:
  # Replace in reverse order so that eg param1 does not interfere with
  # param12 via a substring match.
  if params:
    for i in range(len(params) - 1, -1, -1):
      query = query.replace(f'@param{i}', str(params[i]))

  return query


def get_hinted_query(query: str, hints: str) -> str:
  return f'{hints} {query}'

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

"""Query plan navigation and manipulation utilties."""

from typing import Any, Optional, Sequence

JSON = Any


def filter_keys(plan: JSON,
                keys_to_remove: Optional[Sequence[str]] = None) -> JSON:
  """Removes extraneous keys from plan JSON to reduce file size.

  Args:
    plan: Explain plan in JSON format.
    keys_to_remove: Keys to remove from the plan.

  Returns:
    A new JSON with the specified keys removed.
  """
  if not keys_to_remove:
    return plan

  if isinstance(plan, list):
    return [filter_keys(p, keys_to_remove) for p in plan]
  elif isinstance(plan, dict):
    new_plan = {}
    for k, v in plan.items():
      if k in keys_to_remove:
        continue
      if isinstance(v, (dict, list)):
        new_plan[k] = filter_keys(v, keys_to_remove)
      else:
        new_plan[k] = v
    return new_plan
  else:
    return plan

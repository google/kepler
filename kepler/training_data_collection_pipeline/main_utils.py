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

"""Main file shared functionality.

Overall, we strive to keep main files simple. There are cases where multiple
main files may need to do something very similar. This file contains the helper
functions that are shared between main files.

Please double check when adding more functionality here to see if the
functionality would be better suited in one of the libraries.
"""

import json
import os
from typing import Any, Dict, List

from absl import logging

# Typing aliases.
JSON = Any


class MissingVerificationDataError(ValueError):
  pass


def get_skip_indices(
    plan_hints: Dict[str, List[Dict[str, str]]], verification_file: str
) -> Dict[str, List[int]]:
  """Get plan indices to skip for each query based on verification failures.

  Args:
   plan_hints: Dict mapping query id to list of hints, each a dict mapping
     "hints" to hint string.
   verification_file: Path to verification failures file.

  Returns:
    Dict mapping query id to list of plan indices to skip.

  Raises:
    MissingVerificationDataError: If the verification data doesn't match
      the provided query or hints.
  """
  query_id_to_skip_indices = {}
  if not verification_file:
    return query_id_to_skip_indices

  with open(verification_file) as f:
    hint_failures = json.load(f)
    for query_id in plan_hints:
      if query_id not in hint_failures:
        raise MissingVerificationDataError(
            f"Query {query_id} not in verification file!"
        )
      query_id_to_skip_indices[query_id] = []
      for i, hint in enumerate(plan_hints[query_id]):
        hint_str = hint["hints"]
        if hint_str not in hint_failures[query_id]:
          raise MissingVerificationDataError(
              f"Missing hint {hint_str} for query {query_id}"
          )
        if hint_failures[query_id][hint_str]:
          query_id_to_skip_indices[query_id].append(i)

  return query_id_to_skip_indices


def print_failure_counts(failure_counts: Dict[str, Dict[str, int]]) -> None:
  """Prints out failure summary."""
  logging.info("Printing positive failure counts:")
  for query_id in sorted(failure_counts):
    plan_failure_count = 0
    for hint in sorted(failure_counts[query_id]):
      hint_fail_count = failure_counts[query_id][hint]
      if hint_fail_count > 0:
        plan_failure_count += 1
        logging.info("%s num failures for %s: %s ", query_id, hint,
                     hint_fail_count)
    logging.info("Query %s failure ratio: %d/%d", query_id, plan_failure_count,
                 len(failure_counts[query_id]))


def print_hint_counts_by_source(query_id_to_counts: Dict[str, JSON]) -> None:
  """Prints out query hint counts summary."""
  logging.info("Printing hints counts by source:")
  for query_id in sorted(query_id_to_counts):
    plan_hints = query_id_to_counts[query_id]
    for i, plan_hint in enumerate(sorted(plan_hints.keys())):
      logging.info("%s: number of suggestions for hint %s from source %s: %s",
                   query_id, i, plan_hints[plan_hint]["source"],
                   plan_hints[plan_hint]["count"])


class HintAccumulator:
  """Accumulates extracted hints across queries and saves them to file."""

  def __init__(self):
    self.query_id_to_counts = {}
    self.query_id_to_plan_hints = {}
    self.query_id_to_params_plan_indices = {}
    self.query_id_to_debug_infos = {}
    self.combined_failure_counts = {}

  def save(self, output_dir: str, plans_output_file: str,
           verification_failures_file: str, plan_index_suffix: str) -> None:
    """Saves the content of the hint accumulator to a set of files.

    Args:
      output_dir: The output_dir for produced files.
      plans_output_file: The file to store the plans as hints.
      verification_failures_file: The file to save verification failures.
      plan_index_suffix: The suffix used for plan index file names.
    """
    verification_file_directory = os.path.join(output_dir, "verification")
    os.makedirs(verification_file_directory, exist_ok=True)
    verification_failures_file = os.path.join(verification_file_directory,
                                              verification_failures_file)
    with open(verification_failures_file, "w") as outfile:
      json.dump(self.combined_failure_counts, outfile)

    with open(os.path.join(output_dir, plans_output_file), "w") as outfile:
      json.dump(self.query_id_to_plan_hints, outfile)

    with open(
        os.path.join(output_dir, plans_output_file[:-5] + plan_index_suffix),
        "w") as outfile:
      json.dump(self.query_id_to_params_plan_indices, outfile)

    with open(
        os.path.join(output_dir, plans_output_file[:-5] + "_debug_infos.json"),
        "w") as outfile:
      json.dump(self.query_id_to_debug_infos, outfile)

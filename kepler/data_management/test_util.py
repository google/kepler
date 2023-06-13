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

"""Shared data to simplify unit tests."""

TEST_SEED = 12345
TEST_QUERY_ID = "q0_0"
QUERY_METADATA = {
    TEST_QUERY_ID: {
        "query":
            "SELECT * FROM t as t WHERE a=@param0 AND b=@param1 AND c=@param2",
        "predicates": [{
            "table": "t",
            "alias": "t",
            "column": "a",
            "operator": "=",
            "data_type": "text",
            "distinct_values": ["first"]
        }, {
            "table": "t",
            "alias": "t",
            "column": "b",
            "operator": "=",
            "data_type": "text",
            "distinct_values": ["a", "b"]
        }, {
            "table": "t",
            "alias": "t",
            "column": "c",
            "operator": "=",
            "data_type": "int",
            "min": 0,
            "max": 2
        }, {
            "table": "t",
            "alias": "t",
            "column": "d",
            "operator": "=",
            "data_type": "float",
            "preprocess_type": "to_timestamp",
            "mean": 0,
            "variance": 1
        }]
    }
}
QUERY_EXECUTION_METADATA = {TEST_QUERY_ID: {"plan_cover": [0, 1, 2]}}
QUERY_EXECUTION_DATA = {
    TEST_QUERY_ID: {
        "first####a####2####1970-01-01": {
            "default":
                0,
            "results": [[{
                "duration_ms": 3
            }, {
                "duration_ms": 2
            }, {
                "duration_ms": 1
            }], [{
                "duration_ms": 3
            }, {
                "duration_ms": 2
            }, {
                "duration_ms": 2
            }], [{
                "duration_ms": 5
            }, {
                "duration_ms": 4
            }, {
                "duration_ms": 3
            }], [{
                "duration_ms": 6
            }, {
                "duration_ms": 6
            }, {
                "duration_ms": 4
            }]]
        },
        "first####a####1####1970-01-02": {
            "default":
                0,
            "results": [[{
                "duration_ms": 33
            }, {
                "duration_ms": 22
            }, {
                "duration_ms": 11
            }], [{
                "duration_ms": 33
            }, {
                "duration_ms": 22
            }, {
                "duration_ms": 22
            }], [{
                "duration_ms": 123
            }, {
                "duration_ms": 45
            }, {
                "duration_ms": 33
            }], [{
                "duration_ms": 61
            }, {
                "duration_ms": 64
            }, {
                "duration_ms": 45
            }]]
        },
        "first####b####0####1970-01-03": {
            "default":
                1,
            "results": [[{
                "duration_ms": 333
            }, {
                "duration_ms": 222
            }, {
                "duration_ms": 222
            }],
                        [{
                            "duration_ms": 333
                        }, {
                            "duration_ms": 222
                        }, {
                            "duration_ms": 111
                        }],
                        [{
                            "duration_ms": 555
                        }, {
                            "duration_ms": 444
                        }, {
                            "duration_ms": 333
                        }], [{
                            "skipped": True
                        }]]
        },
        "first####b####1####1970-01-04": {
            "default":
                0,
            "results": [[{
                "duration_ms": 3333
            }, {
                "duration_ms": 3333
            }, {
                "duration_ms": 3333
            }],
                        [{
                            "duration_ms": 3333
                        }, {
                            "duration_ms": 2222
                        }, {
                            "duration_ms": 2222
                        }],
                        [{
                            "duration_ms": 3333
                        }, {
                            "duration_ms": 2222
                        }, {
                            "duration_ms": 1111
                        }], [{
                            "skipped": True
                        }]]
        },
        "first####c####0####1970-01-05": {
            "default":
                2,
            "results": [[{
                "duration_ms": 50
            }, {
                "timed_out": 31
            }, {
                "duration_ms": 15
            }], [{
                "timed_out": 32
            }, {
                "timed_out": 32
            }, {
                "timed_out": 32
            }], [{
                "duration_ms": 3
            }, {
                "duration_ms": 2
            }, {
                "duration_ms": 1
            }], [{
                "skipped": True
            }]]
        },
        "first####c####0####1975-01-05": {
            "default": 2,
            "results": {
                "default_timed_out": 3600
            }
        }
    }
}

QUERY_EXPLAIN_DATA = {
    TEST_QUERY_ID: {
        "first####a####2####1970-01-01": {
            "results": [[{
                "total_cost": 3.1
            }], [{
                "total_cost": 4.1
            }], [{
                "total_cost": 5.1
            }], [{
                "total_cost": 6.1
            }]]
        },
        "first####a####1####1970-01-02": {
            "results": [[{
                "total_cost": 13.1
            }], [{
                "total_cost": 14.1
            }], [{
                "total_cost": 15.1
            }], [{
                "total_cost": 16.1
            }]]
        },
        "first####b####0####1970-01-03": {
            "results": [[{
                "total_cost": 23.1
            }], [{
                "total_cost": 24.1
            }], [{
                "total_cost": 25.1
            }], [{
                "total_cost": 26.1
            }]]
        },
        "first####b####1####1970-01-04": {
            "results": [[{
                "total_cost": 33.1
            }], [{
                "total_cost": 34.1
            }], [{
                "total_cost": 35.1
            }], [{
                "total_cost": 36.1
            }]]
        },
        "first####c####0####1970-01-05": {
            "results": [[{
                "total_cost": 43.1
            }], [{
                "total_cost": 44.1
            }], [{
                "total_cost": 45.1
            }], [{
                "total_cost": 46.1
            }]]
        }
    }
}

PARAMETERS_POOL = [["first", "a", "2", "1970-01-01"],
                   ["first", "a", "1", "1970-01-02"],
                   ["first", "b", "0", "1970-01-03"],
                   ["first", "b", "1", "1970-01-04"],
                   ["first", "c", "0", "1970-01-05"]]

# See README regarding how to run tests.

# pytype binary, test, library

package(default_visibility = ["//kepler:__subpackages__"])

py_library(
    name = "e2e_evaluation",
    srcs = ["e2e_evaluation.py"],
    srcs_version = "PY3",
    deps = [
        "//kepler/data_management:workload",
        # Implicit absl.app dependency.
        # Implicit absl.logging dependency.
    ],
)

py_test(
    name = "e2e_evaluation_test",
    srcs = ["e2e_evaluation_test.py"],
    python_version = "PY3",
    deps = [
        ":e2e_evaluation",
        "//kepler/data_management:test_util",
        "//kepler/data_management:workload",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_binary(
    name = "e2e_evaluation_main",
    srcs = ["e2e_evaluation_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":e2e_evaluation",
        "//kepler/data_management:workload",
        "//kepler/training_data_collection_pipeline:query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

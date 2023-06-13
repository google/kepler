# pytype binary, test, library  # @unused.

package(default_visibility = ["//kepler:__subpackages__"])

py_library(
    name = "database_simulator",
    srcs = ["database_simulator.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit numpy dependency.
        # Implicit pandas dependency.
    ],
)

py_test(
    name = "database_simulator_test",
    srcs = ["database_simulator_test.py"],
    python_version = "PY3",
    deps = [
        ":database_simulator",
        ":test_util",
        ":workload",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_library(
    name = "workload",
    srcs = ["workload.py"],
    srcs_version = "PY3",
    deps = [
        ":database_simulator",
    ],
)

py_test(
    name = "workload_test",
    srcs = ["workload_test.py"],
    python_version = "PY3",
    deps = [
        ":test_util",
        ":workload",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
)

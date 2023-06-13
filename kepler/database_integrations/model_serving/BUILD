# Tests only run on GCP instance. They assume pglast is installed.

# pytype binary, test, library

package(default_visibility = ["//kepler:__subpackages__"])

py_library(
    name = "query_parsing_utils",
    srcs = ["query_parsing_utils.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
)

py_test(
    name = "query_parsing_utils_test",
    srcs = ["query_parsing_utils_test.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":query_parsing_utils",
    ],
)

py_library(
    name = "model_server",
    srcs = ["model_server.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":query_parsing_utils",
        "//kepler/model_trainer:model_base",
    ],
)

py_test(
    name = "model_server_test",
    srcs = ["model_server_test.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":model_server",
        "//kepler/training_data_collection_pipeline:query_utils",
        "//kepler/training_data_collection_pipeline:test_util",
        "//kepler/training_data_collection_pipeline:utils",
        # Implicit numpy dependency.
    ],
)

py_binary(
    name = "model_server_main",
    srcs = ["model_server_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":model_server",
        ":query_parsing_utils",
        "//kepler/model_trainer:sngp_multihead_model",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

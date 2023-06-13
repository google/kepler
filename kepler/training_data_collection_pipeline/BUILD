# See README regarding how to run tests.

# pytype binary, test, library

package(default_visibility = ["//kepler:__subpackages__"])

py_library(
    name = "query_utils",
    srcs = ["query_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":query_text_utils",
        # Implicit absl.logging dependency.
        # Implicit psycopg2 dependency
    ],
)

py_library(
    name = "query_text_utils",
    srcs = ["query_text_utils.py"],
    srcs_version = "PY3",
    deps = [],
)

py_test(
    name = "query_text_utils_test",
    srcs = ["query_text_utils_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":query_text_utils",
        ":test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        ":query_utils",
        # Implicit psycopg2 dependency
    ],
)

py_test(
    name = "query_utils_test",
    srcs = ["query_utils_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":query_utils",
        ":test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_library(
    name = "query_plan_utils",
    srcs = ["query_plan_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "main_utils",
    srcs = ["main_utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
    ],
)

py_test(
    name = "query_plan_utils_test",
    srcs = ["query_plan_utils_test.py"],
    data = glob(["testdata/*"]),
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":query_plan_utils",
        "//pyglib:resources",
        # Implicit absl.testing dependency.
    ],
)

py_library(
    name = "pg_generate_plan_candidates",
    srcs = ["pg_generate_plan_candidates.py"],
    srcs_version = "PY3",
    deps = [
        ":pg_perturb_plan_cardinalities",
        ":pg_plan_hint_extractor",
        ":query_plan_utils",
        ":query_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "pg_generate_plan_candidates_test",
    srcs = ["pg_generate_plan_candidates_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":pg_generate_plan_candidates",
        ":pg_plan_hint_extractor",
        ":query_utils",
        ":test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit numpy dependency.
    ],
)

py_binary(
    name = "pg_generate_plan_candidates_main",
    srcs = ["pg_generate_plan_candidates_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":main_utils",
        ":pg_generate_plan_candidates",
        ":pg_plan_hint_extractor",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
    ],
)

py_library(
    name = "pg_plan_hint_extractor",
    srcs = ["pg_plan_hint_extractor.py"],
    srcs_version = "PY3",
    deps = [
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.logging dependency.
    ],
)

py_test(
    name = "pg_plan_hint_extractor_test",
    srcs = ["pg_plan_hint_extractor_test.py"],
    data = glob(["testdata/*"]),
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":pg_plan_hint_extractor",
        ":query_utils",
        ":test_util",
        "//pyglib:resources",
        # Implicit absl.testing dependency.
    ],
)

py_binary(
    name = "pg_plan_hint_extractor_main",
    srcs = ["pg_plan_hint_extractor_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":main_utils",
        ":pg_plan_hint_extractor",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
    ],
)

py_library(
    name = "pg_execute_training_data_queries",
    srcs = ["pg_execute_training_data_queries.py"],
    srcs_version = "PY3",
    deps = [
        ":query_text_utils",
        ":query_utils",
        "//kepler/data_management:database_simulator",
        # Implicit absl.app dependency.
        # Implicit absl.logging dependency.
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "pg_execute_training_data_queries_test",
    srcs = ["pg_execute_training_data_queries_test.py"],
    python_version = "PY3",
    deps = [
        ":pg_execute_training_data_queries",
        ":query_utils",
        ":test_util",
        "//kepler/data_management:database_simulator",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_binary(
    name = "pg_execute_training_data_queries_main",
    srcs = ["pg_execute_training_data_queries_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":main_utils",
        ":pg_execute_training_data_queries",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

py_binary(
    name = "pg_distributed_training_execution_main",
    srcs = ["pg_distributed_training_execution_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":main_utils",
        ":pg_execute_training_data_queries",
        ":query_text_utils",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

py_binary(
    name = "query_printer_main",
    srcs = ["query_printer_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":query_text_utils",
        # Implicit gfile dependency.
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

py_library(
    name = "parameter_generator",
    srcs = ["parameter_generator.py"],
    srcs_version = "PY3",
    deps = [":query_utils"],
)

py_test(
    name = "parameter_generator_test",
    srcs = ["parameter_generator_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":parameter_generator",
        ":query_utils",
        ":test_util",
        # Implicit absl.testing dependency.
    ],
)

py_binary(
    name = "parameter_generator_main",
    srcs = ["parameter_generator_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parameter_generator",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

py_library(
    name = "pg_perturb_plan_cardinalities",
    srcs = ["pg_perturb_plan_cardinalities.py"],
    srcs_version = "PY3",
    deps = [
        ":pg_execute_training_data_queries",
        ":pg_plan_hint_extractor",
        ":query_plan_utils",
        ":query_utils",
    ],
)

py_test(
    name = "pg_perturb_plan_cardinalities_test",
    srcs = ["pg_perturb_plan_cardinalities_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":pg_perturb_plan_cardinalities",
        ":query_utils",
        ":test_util",
        # Implicit absl.testing dependency.
    ],
)

py_binary(
    name = "pg_generate_plan_costs_main",
    srcs = ["pg_generate_plan_costs_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pg_perturb_plan_cardinalities",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

py_library(
    name = "pg_execute_explain_tools",
    srcs = ["pg_execute_explain_tools.py"],
    srcs_version = "PY3",
    deps = [
        ":pg_plan_hint_extractor",
        ":query_plan_utils",
        ":query_text_utils",
        ":query_utils",
    ],
)

py_test(
    name = "pg_execute_explain_tools_test",
    srcs = ["pg_execute_explain_tools_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":pg_execute_explain_tools",
        ":pg_plan_hint_extractor",
        ":query_utils",
        ":test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_binary(
    name = "pg_execute_explain_tools_main",
    srcs = ["pg_execute_explain_tools_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pg_execute_explain_tools",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
    ],
)

py_test(
    name = "e2e_test",
    srcs = ["e2e_test.py"],
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":parameter_generator",
        ":pg_execute_training_data_queries",
        ":pg_generate_plan_candidates",
        ":pg_plan_hint_extractor",
        ":query_utils",
        ":test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
    ],
)

py_library(
    name = "graph_tensor_builder",
    srcs = ["graph_tensor_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":query_text_utils",
        ":query_utils",
        "//pyglib:resources",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:workload",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow-gnn dependency.
    ],
)

py_test(
    name = "graph_tensor_builder_test",
    srcs = ["graph_tensor_builder_test.py"],
    data = [
        "query_plan_graph_schema.pbtxt",
    ] + glob([
        "testdata/*",
    ]),
    python_version = "PY3",
    tags = [
        "notap",
    ],
    deps = [
        ":graph_tensor_builder",
        ":query_utils",
        ":test_util",
        "//pyglib:resources",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow-gnn dependency.
    ],
)

py_binary(
    name = "graph_tensor_builder_main",
    srcs = ["graph_tensor_builder_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_tensor_builder",
        ":query_utils",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit tensorflow dependency.
    ],
)

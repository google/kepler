# pytype binary, test, library

package(default_visibility = ["//kepler:__subpackages__"])

py_library(
    name = "trainer_util",
    srcs = ["trainer_util.py"],
    deps = [
        "//kepler/data_management:workload",
        # Implicit python-dateutil dependency.
        # Implicit numpy dependency.
        # Implicit pandas dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "trainer_util_test",
    srcs = ["trainer_util_test.py"],
    deps = [
        ":evaluation",
        ":test_util",
        ":trainer_util",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:test_util",
        "//kepler/data_management:workload",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        ":model_base",
        ":trainer_util",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:workload",
        # Implicit absl.logging dependency.
        # Implicit numpy dependency.
        # Implicit pandas dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    deps = [
        ":model_base",
        ":multihead_model",
        ":test_util",
        ":trainer",
        ":trainer_util",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:test_util",
        "//kepler/data_management:workload",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "trainer_main",
    srcs = ["trainer_main.py"],
    deps = [
        ":evaluation",
        ":model_base",
        ":sngp_multihead_model",
        ":trainer",
        ":trainer_util",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:workload",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":trainer_util",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:workload",
        # Implicit absl.logging dependency.
        # Implicit numpy dependency.
        # Implicit scipy dependency.
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    deps = [
        ":evaluation",
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:test_util",
        "//kepler/data_management:workload",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "loss_functions",
    srcs = ["loss_functions.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "loss_functions_test",
    srcs = ["loss_functions_test.py"],
    deps = [
        ":loss_functions",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        ":loss_functions",
        ":model_base",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "model_base",
    srcs = ["model_base.py"],
    deps = [
        ":trainer_util",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "model_base_test",
    srcs = ["model_base_test.py"],
    deps = [
        ":model_base",
        ":test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "multihead_model",
    srcs = ["multihead_model.py"],
    deps = [
        ":model_base",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "sngp_multihead_model",
    srcs = ["sngp_multihead_model.py"],
    deps = [
        ":model_base",
        ":multihead_model",
        ":trainer_util",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow_models dependency.
    ],
)

py_test(
    name = "sngp_multihead_model_test",
    srcs = ["sngp_multihead_model_test.py"],
    deps = [
        ":model_base",
        ":sngp_multihead_model",
        ":test_util",
        ":trainer_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "multihead_model_test",
    srcs = ["multihead_model_test.py"],
    deps = [
        ":model_base",
        ":multihead_model",
        ":sngp_multihead_model",
        ":test_util",
        ":trainer_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

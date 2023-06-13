# pytype binary

package(default_visibility = ["//kepler:__subpackages__"])

py_binary(
    name = "active_learning_for_training_data_collection_main",
    srcs = ["active_learning_for_training_data_collection_main.py"],
    deps = [
        "//kepler/data_management:database_simulator",
        "//kepler/data_management:workload",
        "//kepler/model_trainer:evaluation",
        "//kepler/model_trainer:model_base",
        "//kepler/model_trainer:sngp_multihead_model",
        "//kepler/model_trainer:trainer",
        "//kepler/model_trainer:trainer_util",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

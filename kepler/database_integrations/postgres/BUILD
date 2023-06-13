# Tests only run on GCP instance. They assume Postgres with the Kepler-extended
# pg_hint_plan is installed.

# pytype test

py_test(
    name = "pg_hint_plan_extension_test",
    srcs = ["pg_hint_plan_extension_test.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//kepler/training_data_collection_pipeline:pg_plan_hint_extractor",
        "//kepler/training_data_collection_pipeline:query_utils",
        "//kepler/training_data_collection_pipeline:test_util",
        # Implicit absl.testing dependency.
        # Implicit absl.parameterized dependency.
        # Implicit psycopg2 dependency
    ],
)

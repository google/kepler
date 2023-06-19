# Kepler

Kepler is a learning-based optimizer for parameterized queries that provides:

1.  faster query execution time,

2.  fast query planning time, and

3.  robustness, i.e. is not worse than the existing optimizer with high
    probability.

Broadly, for a given query template, Kepler generates a set of candidate plans
via Row Count Evolution (RCE), then executes those plans on a workload of query
instances. This execution data is then used as a training dataset for machine
learning models that predict the best plan for any given query instance. Kepler
leverages Spectral-Normalized Gaussian Process (SNGP) models to produce
calibrated confidence scores for its predictions, and will fall back to the
built-in (Postgres) optimizer if its confidence score is not sufficiently high.

For more details, see our paper, [Kepler: Robust Learning for Parametric Query
Optimization](http://arxiv.org/abs/2306.06798), to appear at SIGMOD 2023.

## Usage

Usage for the individual library components can be found in their respective
comment headers.

The examples directory includes a demonstration for how one could use the Kepler
data set and DatabaseSimulator for active learning research to reduce the
training data collection cost of Kepler. A sample run command is provided in the
[Build and Run](#build-and-run) section below.

## Dataset

To benchmark Kepler, we constructed a dataset using StackExchange data, which we
call the Stack Parametric Query Dataset (SPQD). SPQD is based on the original
[Stack](https://rm.cab/stack) dataset, and consists of 87 unique query
templates. For each template, up to 50000 synthetic query instances are
provided, and a set of candidate plans (generated via RCE) are executed over
this workload, yielding nearly 14 years worth of execution data.

Using SPQD enables research on the machine learning aspects of the problem
without requiring a live database instance or paying for the cost of additional
query execution. SPQD contains the cross-product of the plan-cover candidate
plans and query instances. This enables a researcher to evaluate approaches to
reduce the required query instance executions to train a sufficient model by
using SPQD to cost and compare counterfactuals.

Of the 87 query templates in SPQD, 42 are automatically extracted from the
original Stack dataset, and the remaining 45 are manually written. Full dataset
details can be found in the paper.

The dataset can be downloaded at
[https://storage.googleapis.com/gresearch/stack_parametric_query_dataset/stack_parametric_query_dataset.zip](https://storage.googleapis.com/gresearch/stack_parametric_query_dataset/stack_parametric_query_dataset.zip)
(2.7 GB).

### Dataset Structure

The base directory with `LICENSE.txt` consists of the following:

1.  `stack_query_templates_with_metadata.json` contains the query templates,
    query instance parameters, and auxiliary metadata.

2.  The `training_metadata` directory contains auxiliary information about
    database table contents that is used in model training (e.g. vocabulary
    selection).

3.  The `execution_data` directory contains a `hints` subdirectory for
    RCE-generated plans and a `results` subdirectory for execution latency data
    and plan-cover metadata. The `results` subdirectory comprises the outputs of
    our training data pipeline code and corresponds directly to the input data
    for model training. The utilities in the `data_management` package are the
    recommended interface for using this data.

## Build and Run

Environment tools:

*   The repository uses the bazel build tool.

*   The workflow has been tested with python3.10 and uses `python3.10-venv`. A
    python version below 3.8 will definitely not work due to the tensorflow
    version.

*   The psycopg2 in `requirements.txt` requires `libpq-dev` (or the equivalent
    for the OS)

*   Some libraries require installing `build-essential`.

*   This implementation expects to connect to a Postgres 13 instance with
    [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan/tree/PG13) for PG13
    installed.

Ubuntu-friendly command:

```
sudo apt-get install python3.10-venv libpq-dev build-essential
```

After installing the requirements, use `bazel build` for building and `bazel
run` or `bazel-bin` to execute. The following sample commands presume `BASE` is
an environment variable set to the base repository directory containing
`README.md` and `requirements.txt`.

```
cd $BASE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bazel build kepler/...
```

To run the active learning example, first download and unzip the
[SPQD dataset](#dataset). The following command presumes `SPQD` is the base
dataset directory containing `LICENSE.txt`.

```
cd $BASE
./bazel-bin/kepler/examples/active_learning_for_training_data_collection_main --query_metadata  $SPQD/stack_query_templates_with_metadata.json  --vocab_data_dir $SPQD/training_metadata --execution_metadata $SPQD/execution_data/results/q31_0/execution_output/stack_q31_0_metadata.json --execution_data $SPQD/execution_data/results/q31_0/execution_output/stack_q31_0.json  --query_id q31_0
```

## Disclaimer

This is not an officially supported Google product.

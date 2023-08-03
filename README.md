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

For more details, see our SIGMOD 2023 paper
[Kepler: Robust Learning for Parametric Query Optimization](https://dl.acm.org/doi/abs/10.1145/3588963).
A brief
[summary](https://www.growkudos.com/publications/10.1145%25252F3588963/reader)
appears on Kudos.

## Usage

Usage for the individual library components can be found in their respective
comment headers.

The `examples` directory source code includes a demonstration for how one could
use the Kepler dataset and associated tooling like `DatabaseSimulator` for
active learning research to reduce the training data collection cost of Kepler.
A sample run command is provided in the
[Using SPQD with DatabaseSimulator](#using-spqd-with-databasesimulator) section
below.

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

## Run

### General Set Up

Environment tools:

*   The workflow has been tested with python3.10 and requires `python3.10-venv`
    and `python3.10-dev`. A python version below 3.8 will definitely not work
    due to the tensorflow version.

*   The psycopg2 in `requirements.txt` requires `libpq-dev` (or the equivalent
    for the OS)

*   Some libraries require installing `build-essential`.

Ubuntu-friendly command:

```
sudo apt-get install python3.10-venv python3.10-dev libpq-dev build-essential
```

The following sample commands presume `BASE` is an environment variable set to
the base repository directory containing `README.md` and `requirements.txt`.

```
cd $BASE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Using SPQD with DatabaseSimulator

To run the active learning example, first download and unzip the
[SPQD dataset](#dataset). The following command presumes `SPQD` is the base
dataset directory containing `LICENSE.txt`.

*Note: No live database connection is required.*

```
cd $BASE
python -m kepler.examples.active_learning_for_training_data_collection_main --query_metadata  $SPQD/stack_query_templates_with_metadata.json  --vocab_data_dir $SPQD/training_metadata --execution_metadata $SPQD/execution_data/results/q31_0/execution_output/stack_q31_0_metadata.json --execution_data $SPQD/execution_data/results/q31_0/execution_output/stack_q31_0.json  --query_id q31_0
```

### Postgres Set Up

Postgres set up is **not required** for machine learning research using the SPQD
and the associated `DatabaseSimulator` tool from the `data_management` package.

The utilities which do connect to a database were tested using a Postgres 13
instance with [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan/tree/PG13)
for PG13 installed. The instructions below cover the installation of
`pg_hint_plan` but do not cover the installation of Postgres 13.

Note that installing `pg_hint_plan` may require first installing Postgres dev
libraries, such as via the following:

```
sudo apt install postgresql-server-dev-13
```

After installing Postgres 13 and the matching version of `pg_hint_plan` as shown
below, execute `CREATE EXTENSION pg_hint_plan` from the `psql` prompt:

```
git clone https://github.com/ossc-db/pg_hint_plan.git
cd pg_hint_plan
git fetch origin PG13
git checkout PG13
sudo make install
sudo service postgresql restart
```

If you have not created a user on Postgres yet, you may need to use `sudo su -
postgres` before typing `psql`.

The pg_stat_statements library needs to be enabled. This is typically done by
adding `pg_stat_statements` to the `shared_preload_libraries` line in
`/etc/postgresql/13/main/postgresql.conf` and then restarting Postgres. The
edited line may look like this:

```
shared_preload_libraries = 'pg_stat_statements'
```

At this point, one can execute training data collection queries and build models
using this codebase. The remaining steps describe additional set up to repeat
how this project integrated Kepler into Postgres at query time. The pg_hint_plan
extension was patched to reach out to a server which hosts the models. The
server checks the hint string in the query for a query id. If the server has a
model matching the query id, it will produce a set of plan hints. See
`kepler/database_integrations/model_serving` for an implementation of this
server.

To patch `pg_hint_plan` for PG13, amend paths below and run the following:

```
path_to_pg_hint_plan="$HOME/pg_hint_plan"
path_to_patch="$HOME/kepler/kepler/database_integrations/postgres/13/kepler_extension.patch"
cd $path_to_pg_hint_plan
patch -p0 < $path_to_patch
sudo make install
sudo service postgresql restart
```

### Running Binaries

*Coming soon.*

### Running Tests

Running tests requires a few additional steps after installing Postgres 13 and
pg_hint_plan. Be sure to have executed `CREATE EXTENSION pg_hint_plan` on the
database, per instructions above. Open a prompt to Postgres by typing `psql` in
the shell. As before, if you have not created a user on Postgres yet, you may
need to use `sudo su - postgres` before typing `psql`. Then execute the
following commands:

```
CREATE USER test SUPERUSER PASSWORD 'test';
CREATE DATABASE test;
```

pytest kepler

```
cd $BASE
pytest kepler
```

## Disclaimer

This is not an officially supported Google product.

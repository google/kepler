# Kepler

Kepler is a learning-based optimizer for parameterized queries that provides: 1.
faster query execution time, 2. fast query planning time, and 3. robustness,
i.e. is never worse than the existing optimizer with high probability.

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
training data collection cost of Kepler.

## Dataset

We will release a dataset containing ~14 years worth of parametric query
execution data over the StackExchange dataset. Stay tuned!

## Disclaimer

This is not an officially supported Google product.

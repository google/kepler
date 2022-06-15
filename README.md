# Kepler

Kepler is a research project employing machine learning to improve database
query execution time for parameterized queries. By doing so, it attempts to
overcome shortcomings in the query optimizer cost model and cardinality
estimates. To learn more about the project please see our paper on
https://arxiv.org.

The Kepler codebase consists of:
1) Training data collector which gathers empirical query execution
2) Database simulator used simulate query execution to get query execution
latency
3) Model trainer to build and train a model to recommend query plans

Usage for the individual library components can be found in their respective
comment headers.

For additional support, please reach out to the Learned Systems team at
learned-systems@google.com.

# Shrinkr - Covariance matrix shrinkage and LDA
<a href="https://jgrzywaczewski.com/shrinkr">
    <img src="docs/shrinkr.svg" align="right" height="139" class="readme-logo" alt="Shrinkr logo">
</a>

[![shrinkr for Python](https://github.com/ZetrextJG/shrinkr/actions/workflows/tests.yml/badge.svg)](https://github.com/ZetrextJG/shrinkr/actions/workflows/tests.yml)

Shrinkr is a Python package for covariance matrix shrinkage and Linear Discriminant Analysis.
Methods are implemented in C for performance and exposed through a clean Python interface.

## Installation

Currently the package is only on GitHub. Install most recent release with:
```sh
pip install git+https://github.com/ZetrextJG/shrinkr@latest
```

> PyPI release coming soon.


## Usage example

Also located in the [ready to run script](https://github.com/ZetrextJG/shrinkr/blob/main/.devel/usage.py).

```python
from shrinkr import CovarianceEstimator
from shrinkr import LinearDiscriminantAnalysis as LDA
from shrinkr.functional import accuracy
from shrinkr.monte_carlo import get_guassian_lda_samples

# Generate Gaussian data for covariance estimation and LDA
X, y = get_guassian_lda_samples(p=20, n_per_class=200, seed=1)

# Shrunk covariance estimation:
# Methods like LW Linear, OAS, LW Analytical.
total_covariance = CovarianceEstimator(method="lw_linear").fit_predict(X)
assert total_covariance.shape == (20, 20)

# Linear Discriminant Analysis with Shrunk covariance estimation:
# Supports all methods from CovarianceEstimator
# but also LDA specialized shrinkages like DEAL.
classifier = LDA(method="deal")
classifier.fit(X, y)
y_pred = classifier.predict(X)
print(accuracy(y, y_pred)) # 0.935, quite a simple task
```

## Documentation

Documentation site is hosted on [GitHub Pages](https://jgrzywaczewski.com/shrinkr/).
Build with [MkDocs](https://www.mkdocs.org/) for Python and [Doxygen](https://www.doxygen.nl/) for C
API Reference.

## Structure

Main classes `CovarianceEstimator` and `LinearDiscriminantAnalysis` are importable
directly from the package root `shrinkr.*`.
All shrinkage methods are implemented functionally in the `shrinkr.functional` module,
with reference Python/NumPy implementations in `shrinkr.reference`.
Additionally Monte Carlo implementations used for tests (and more) 
are located in the `shrinkr.monte_carlo`.

## Development

The project is set up with [uv](https://docs.astral.sh/uv/):
```sh
uv sync --dev
```

The pure C code can be found in `./src` with the Python
bindings in `./shrinkr/bindings.c` which are exposed via the
`shrinkr._native` module with type interface in `./shrinkr/_native.pyi`.

### Testing

All tests are in `./.devel/tests` and are handled with [pytest](https://docs.pytest.org/en/stable/).

To run the unit test suite:
```sh
uv run pytest -m unit
```
Unit tests are designed to run in under a second and execute on every commit.

To run the property-based test suite:
```sh
uv run pytest -m prop
```
Property-based tests cover a wide range of inputs and are expected to run before releases.

### Styling

Styling is handled entirely with [ruff](https://docs.astral.sh/ruff/) and enforced on every commit by [pre-commit](https://pre-commit.com/).
Docstrings must be in the [numpy docstring](https://numpy.org/doc/1.19/docs/howto_document.html)
format. Also enforced by *ruff*.

### Benchmarking

Benchmarking tools can be found in `./.devel/bench`.
Those utilize [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark) for
benchmarking together with Python wrappers and [Google's benchmark](https://github.com/google/benchmark)
for the benchmarking pure C implementation.

## See also

Other amazing projects from which I took motivation:
- [scikit-learn](https://github.com/scikit-learn/scikit-learn): Machine Learning in Python
- [deadwood](https://github.com/gagolews/deadwood): Outlier Detection via Pruning Mutual Reachability Minimum Spanning Trees
- [gips](https://github.com/PrzeChoj/gips): Gaussian model Invariant by Permutation Symmetry


## Benchmark results

Benchmarking results run on a Lenovo ThinkSystem SR665 with 2x AMD EPYC 7413 48 Core Processors and sufficient RAM. 
The number of cores is restricted to 16. *Numpy* was installed with *uv*.

| Estimator          | Method   | Time (ms)   | Rounds   | Diff vs Numpy   | Ratio   |
|:-------------------|:---------|:------------|:---------|:----------------|:--------|
| **p=1000, n=2000** |          |             |          |                 |         |
| LW_Analytical      | shrinkr  | 0.389       | 2427     | - 33.696 ms     |         |
|                    | Numpy    | 34.085      | 35       | + 33.696 ms     | 87.54x  |
| LW_Linear          | shrinkr  | 28.751      | 48       | - 280.091 ms    |         |
|                    | Numpy    | 308.842     | 5        | + 280.091 ms    | 10.74x  |
| OAS                | shrinkr  | 2.591       | 551      | + 1.892 ms      |         |
|                    | Numpy    | 0.699       | 634      | - 1.892 ms      | 0.27x   |
| **p=500, n=1000**  |          |             |          |                 |         |
| LW_Analytical      | shrinkr  | 0.145       | 7010     | - 3.825 ms      |         |
|                    | Numpy    | 3.970       | 87       | + 3.825 ms      | 27.41x  |
| LW_Linear          | shrinkr  | 41.990      | 21       | - 86.015 ms     |         |
|                    | Numpy    | 128.005     | 9        | + 86.015 ms     | 3.05x   |
| OAS                | shrinkr  | 0.891       | 1355     | + 0.699 ms      |         |
|                    | Numpy    | 0.193       | 1608     | - 0.699 ms      | 0.22x   |
| **p=210, n=300**   |          |             |          |                 |         |
| LW_Analytical      | shrinkr  | 0.074       | 15994    | - 1.306 ms      |         |
|                    | Numpy    | 1.380       | 691      | + 1.306 ms      | 18.56x  |
| LW_Linear          | shrinkr  | 55.001      | 15       | - 9.992 ms      |         |
|                    | Numpy    | 64.993      | 24       | + 9.992 ms      | 1.18x   |
| OAS                | shrinkr  | 0.399       | 4602     | + 0.358 ms      |         |
|                    | Numpy    | 0.041       | 4642     | - 0.358 ms      | 0.10x   |
| **p=190, n=300**   |          |             |          |                 |         |
| LW_Analytical      | shrinkr  | 0.053       | 16638    | - 1.134 ms      |         |
|                    | Numpy    | 1.187       | 783      | + 1.134 ms      | 22.21x  |
| LW_Linear          | shrinkr  | 46.904      | 17       | + 42.084 ms     |         |
|                    | Numpy    | 4.820       | 198      | - 42.084 ms     | 0.10x   |
| OAS                | shrinkr  | 0.342       | 26       | + 0.306 ms      |         |
|                    | Numpy    | 0.035       | 14436    | - 0.306 ms      | 0.10x   |
| **p=70, n=60**     |          |             |          |                 |         |
| LW_Analytical      | shrinkr  | 0.033       | 21160    | - 0.140 ms      |         |
|                    | Numpy    | 0.173       | 3702     | + 0.140 ms      | 5.27x   |
| LW_Linear          | shrinkr  | 0.129       | 8571     | + 0.041 ms      |         |
|                    | Numpy    | 0.088       | 9395     | - 0.041 ms      | 0.68x   |
| OAS                | shrinkr  | 0.086       | 38       | + 0.072 ms      |         |
|                    | Numpy    | 0.014       | 39066    | - 0.072 ms      | 0.16x   |
| **p=50, n=60**     |          |             |          |                 |         |
| LW_Analytical      | shrinkr  | 0.025       | 28600    | - 0.186 ms      |         |
|                    | Numpy    | 0.211       | 2659     | + 0.186 ms      | 8.41x   |
| LW_Linear          | shrinkr  | 0.082       | 11524    | + 0.016 ms      |         |
|                    | Numpy    | 0.066       | 11240    | - 0.016 ms      | 0.81x   |
| OAS                | shrinkr  | 0.056       | 20326    | + 0.043 ms      |         |
|                    | Numpy    | 0.013       | 37270    | - 0.043 ms      | 0.23x   |


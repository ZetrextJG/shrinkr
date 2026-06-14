# Shrinkr - Covariance matrix shrinkage and LDA

[![shrinkr for Python](https://github.com/ZetrextJG/shrinkr/actions/workflows/tests.yml/badge.svg)](https://github.com/ZetrextJG/shrinkr/actions/workflows/tests.yml)

Shrinkr is a Python package which implements common covariance
matrix shrinkage methods written in C and exposes them via a Python interface.

## Installation

Currently the package is only on GitHub.
`pip install git+https://github.com/ZetrextJG/shrinkr`

> The PyPI coming soon.


## Usage example

```python
from shrinkr import CovarianceEstimator
from shrinkr import LinearDiscriminantAnalysis as LDA
from shrinkr.functional import accuracy
from shrinkr.monte_carlo import get_guassian_lda_samples

# Generate Guassian data for covariance estimation and LDA
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
print(accuracy(y, y_pred)) # 1.0, quite a simple task
```

## Documentation

Documentation site is hosted on [GitHub Pages](https://jgrzywaczewski.com/shrinkr/).

## Structure

All implemented methods are implemented functionally (`shrinkr.functional` module)
have the reference implementations in Python (Numpy)
which can be found in `shrinkr.reference` module.


## Development

The project is set up with `uv`:
```
uv sync --dev
```

The pure C code can be found in `./src` with the Python
bindings in `./shrinkr/bindings.c` which are exposed via the
`shrinkr._native` module with type interface in `./shrinkr/_native.pyi`.


### Testing

All tests are in `./.devel/tests` and are handled with *pytest*.

To run the unit test suite, run the following command in the terminal:
```
uv run pytest -m unit
```
Unit tests are designed to take less then a second to run 
and run on every commit to ensure that the code is working as expected.

To run the property-based test suite, run the following command in the terminal:
```
uv run pytest -m prop
```
Property based test are designed to check the functionality of the code on a wide range of inputs
and are expected to run before releases.

### Styling

Styling is handled entirely with *ruff* and enforced on every commit by *pre-commit*.
Docstrings must be in the [numpy docstring](https://numpy.org/doc/1.19/docs/howto_document.html)
format. Also enforced by *ruff*.


### Benchmarking
Benchmarking tools can be found in `./.devel/bench`.


## Benchmark results

Benchmarking results run on a Lenovo ThinkSystem SR665 with 2x AMD EPYC 7413 48 Core Processors and sufficient RAM. 
The number of cores is restricted to 16. Numpy is installed with uv.

...


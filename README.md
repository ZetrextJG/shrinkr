# Shrinkr - Covariance matrix shrinkage and LDA

Shrinkr is a Python package which implements common covariance
matrix shrinkage methods written in C and exposes them via a Python interface.

## Installation

Currently the package is only on GitHub.
`pip install git+https://github.com/ZetrextJG/shrinkr`

> The PyPI implementation will be available soon. 


## Usage

> TODO

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


### Tests

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

### Benchmarking

Benchmarking tools can be found in `./.devel/bench`.

### Styling

Styling is handled entirely with *ruff* and enforced on every commit by *pre-commit*.


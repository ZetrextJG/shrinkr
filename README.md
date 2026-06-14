# Shrinkr - Covariance matrix shrinkage and LDA

[![shrinkr for Python](https://github.com/ZetrextJG/shrinkr/actions/workflows/tests.yml/badge.svg)](https://github.com/ZetrextJG/shrinkr/actions/workflows/tests.yml)

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
Docstrings must be in the [numpy docstring](https://numpy.org/doc/1.19/docs/howto_document.html)
format. Also enforced by *ruff*.


## Benchmark results (8 cores and 16 cores)

Benchmarking results run on a Lenovo ThinkSystem SR665 with 2x AMD EPYC 7413 48 Core Processors and sufficient RAM. 

| Algorithm Group | Impl | `p` | `n` | Median (8 Cores) | Rounds (8 Cores) | Median (16 Cores) | Rounds (16 Cores) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LW_Analytical** | `C+OpenMP` | 50 | 60 | 23.8940 us | 28,941 | 23.8530 us | 28,706 |
|  | `Numpy` | 50 | 60 | 274.2885 us | 1,870 | 207.2277 us | 2,408 |
|  | `C+OpenMP` | 70 | 60 | 31.3185 us | 24,941 | 31.2198 us | 24,448 |
|  | `Numpy` | 70 | 60 | 229.1249 us | 2,592 | 240.8922 us | 3,063 |
|  | `C+OpenMP` | 180 | 300 | 53.1990 us | 46 | 45.7447 us | 19,923 |
|  | `Numpy` | 180 | 300 | 506.5184 us | 1,706 | 503.4916 us | 1,598 |
|  | `C+OpenMP` | 220 | 300 | 79.9596 us | 13,295 | 60.3925 us | 17,625 |
|  | `Numpy` | 220 | 300 | 716.7645 us | 1,285 | 707.5164 us | 1,271 |
|  | `C+OpenMP` | 500 | 600 | 265.1215 us | 3,654 | 139.9517 us | 6,197 |
|  | `Numpy` | 500 | 600 | 10,580.4000 us | 94 | 10,074.8362 us | 92 |
|  | `C+OpenMP` | 700 | 600 | 276.0934 us | 3,498 | 167.5999 us | 6,480 |
|  | `Numpy` | 700 | 600 | 6,931.3440 us | 157 | 8,736.5508 us | 123 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LW_Linear** | `C+OpenMP` | 50 | 60 | 150.1897 us | 6,112 | 153.5583 us | 6,040 |
|  | `Numpy` | 50 | 60 | 64.6915 us | 10,969 | 66.3120 us | 10,568 |
|  | `C+OpenMP` | 70 | 60 | 281.2762 us | 3,507 | 288.1754 us | 3,506 |
|  | `Numpy` | 70 | 60 | 87.2128 us | 9,385 | 87.7231 us | 9,460 |
|  | `C+OpenMP` | 180 | 300 | 8.8142 ms | 115 | 8.8149 ms | 72 |
|  | `Numpy` | 180 | 300 | 2.9842 ms | 361 | 2.7234 ms | 351 |
|  | `C+OpenMP` | 220 | 300 | 2.7174 ms | 317 | 2.5348 ms | 461 |
|  | `Numpy` | 220 | 300 | 3.4235 ms | 293 | 3.0798 ms | 44 |
|  | `C+OpenMP` | 500 | 600 | 20.7751 ms | 54 | 14.4325 ms | 69 |
|  | `Numpy` | 500 | 600 | 7.7892 ms | 123 | 19.4928 ms | 39 |
|  | `C+OpenMP` | 700 | 600 | 38.0859 ms | 26 | 23.4456 ms | 46 |
|  | `Numpy` | 700 | 600 | 12.9737 ms | 72 | 10.3626 ms | 88 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **OAS** | `C+OpenMP` | 50 | 60 | 48.9037 us | 25,705 | 48.7920 us | 22,261 |
|  | `Numpy` | 50 | 60 | 12.0811 us | 39,160 | 12.1128 us | 28,275 |
|  | `C+OpenMP` | 70 | 60 | 93.1257 us | 14,470 | 40.4362 us | 11,194 |
|  | `Numpy` | 70 | 60 | 13.7761 us | 41,260 | 13.9773 us | 38,527 |
|  | `C+OpenMP` | 180 | 300 | 252.6212 us | 5,025 | 402.5642 us | 2,727 |
|  | `Numpy` | 180 | 300 | 34.0324 us | 21,103 | 33.2016 us | 16,154 |
|  | `C+OpenMP` | 220 | 300 | 351.7447 us | 3,524 | 517.5183 us | 1,922 |
|  | `Numpy` | 220 | 300 | 46.8232 us | 16,388 | 43.3419 us | 3,542 |
|  | `C+OpenMP` | 500 | 600 | 673.0016 us | 1,558 | 1,069.6929 us | 1,043 |
|  | `Numpy` | 500 | 600 | 182.6435 us | 3,414 | 161.4504 us | 2,095 |
|  | `C+OpenMP` | 700 | 600 | 3,519.1160 us | 648 | 2,024.9039 us | 453 |
|  | `Numpy` | 700 | 600 | 345.3474 us | 1,387 | 302.7227 us | 1,706 |


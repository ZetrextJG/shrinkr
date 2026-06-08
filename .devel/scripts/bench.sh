#!/bin/bash

uv pip install -e .


export N_CORES=8
export OPENBLAS_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OMP_NUM_THREADS=$N_CORES
export VECLIB_MAXIMUM_THREADS=$N_CORES
export NUMEXPR_NUM_THREADS=$N_CORES

OMP_WAIT_POLICY=ACTIVE uv run pytest ./.devel/bench/pytest_bench_methods.py

#!/bin/bash

# Fist cleanup the build artifacts
./.devel/scripts/cleanup.sh

uv sync --all-groups
uv pip install -e .

uv run ./.devel/bench/data/gen_data.py



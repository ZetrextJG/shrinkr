#!/bin/bash

# Fist cleanup the build artifacts
./.devel/scripts/cleanup.sh

uv sync --all-groups
uv pip install -e .



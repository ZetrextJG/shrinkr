#!/bin/bash

# rm -rf docs/_build/html

uv sync --group docs

uv pip install -e .

uv run mkdocs serve


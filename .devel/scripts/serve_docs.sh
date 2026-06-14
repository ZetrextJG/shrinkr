#!/bin/bash

# rm -rf docs/_build/html

uv sync --group docs

uv pip install -e .

cp README.md ./docs/index.md
uv run mkdocs serve


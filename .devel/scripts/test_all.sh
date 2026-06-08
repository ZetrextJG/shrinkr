#!/bin/bash

uv pip install -e .
uv run pytest -s

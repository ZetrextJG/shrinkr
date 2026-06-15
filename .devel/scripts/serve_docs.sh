#!/bin/bash

# Cleanup
rm -rf site

uv sync --group docs

uv pip install -e .

cp README.md ./docs/index.md
sed -i "s/docs\/shrinkr.svg/shrinkr.svg/g" docs/index.md
cp NEWS.md ./docs/news.md

uv run mkdocs serve


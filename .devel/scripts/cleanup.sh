#!/bin/bash

rm -rf .pytest_cache
rm -rf .hypothesis
rm -rf .benchmarks
rm -rf .ruff_cache
rm -rf *.egg-info

find src -type f -name "*.so" -delete
find shrinkr -type f -name "*.so" -delete

find src -type f -name "*.o" -delete
find shrinkr -type f -name "*.o" -delete

find shrinkr -type f -name "*.pyd" -delete
find shrinkr -type d -name "__pycache__" -exec rm -rf {} +

find .devel -type f -name "*.pyd" -delete
find .devel -type d -name "__pycache__" -exec rm -rf {} +

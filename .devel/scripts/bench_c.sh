#!/bin/bash

set -e

cd ./src/

gcc -O3 -march=native -flto -fopenmp -c c_shrinkr.c
g++ -O3 -march=native -flto -fopenmp benchmark.cpp c_shrinkr.o -lbenchmark -lpthread
./a.out

rm ./a.out

cd ..


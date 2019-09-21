#!/bin/sh

rm -r -f build
mkdir build
cd build
cmake ..
make -j 8
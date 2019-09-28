#!/bin/sh

if [ "$1" = "-d" ]
then
    echo "Building debug"
    FOLDER_NAME="debug"
    BUILD_COMMAND="cmake -DCMAKE_BUILD_TYPE=Debug -D HAS_MATPLOTLIB_LIB=ON .."
else
    echo "Building release"
    FOLDER_NAME="build"
    BUILD_COMMAND="cmake -D HAS_MATPLOTLIB_LIB=ON .."
fi

rm -r -f $FOLDER_NAME
mkdir $FOLDER_NAME
cd $FOLDER_NAME
$BUILD_COMMAND
make -j 8

cd tests
ctest
#!/bin/sh

if [ "$1" = "-d" ]
then
    echo "Building debug"
    FOLDER_NAME="debug"
    BUILD_COMMAND="cmake -DCMAKE_BUILD_TYPE=Debug .."
else
    echo "Building release"
    FOLDER_NAME="build"
    BUILD_COMMAND="cmake .."
fi

rm -r -f $FOLDER_NAME
mkdir $FOLDER_NAME
cd $FOLDER_NAME
$BUILD_COMMAND
make -j 8

cd tests
ctest
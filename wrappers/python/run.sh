#!/bin/bash

if [ "$#" == 0 ]
then
    ./clean.sh && swig -python -py3 c3.i && python setup.py build_ext --inplace > py_build_msg.log
else 
    if [ "$1" == "3" ]
    then
        echo "Building with Python 3.x" > py_build_msg.log
        ./clean.sh && swig -python -py3 c3.i && python setup.py build_ext --inplace >> py_build_msg.log
    elif [ "$1" == "2" ]
    then
        echo "Building with Python 2.x" > py_build_msg.log 
        ./clean.sh && swig -python c3.i && python setup.py build_ext --inplace >> py_build_msg.log
    else
        echo "Unknown Python Version $1"
    fi
fi

#!/bin/bash

./clean.sh && swig -python -py3 c3.i && python setup.py build_ext --inplace

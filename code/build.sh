#!/bin/bash
#rm lib/*.so
rm -f *.so
python ./setup.py build_ext --inplace

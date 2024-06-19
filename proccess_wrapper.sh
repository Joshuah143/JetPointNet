#!/bin/bash

export ROOT_FILE_0=$1
export ROOT_FILE_1=$2
export ROOT_FILE_2=$3
export ROOT_FILE_3=$4
export ROOT_FILE_4=$5
export USING_GRID=1

source /cvmfs/sft.cern.ch/lcg/views/dev4/latest/x86_64-el9-gcc13-opt/setup.sh

python3 python_scripts/data_processing/jets/jets_root_to_awk.py

python3 python_scripts/data_processing/jets/jets_awk_to_npz.py
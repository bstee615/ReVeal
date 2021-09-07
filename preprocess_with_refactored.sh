#!/bin/bash

name="chrome_debian"
inpdir="data/$name"
outpdir="out/${name}_refactored"

mkdir -p "$outpdir"
mkdir -p "${inpdir}_refactored/"

export PYTHONPATH="$PWD:$PYTHONPATH"
python data_processing/preprocess.py --project "$name" --load "out/${name}/" --input "${inpdir}_refactored/" --output "$outpdir/" || exit 1;

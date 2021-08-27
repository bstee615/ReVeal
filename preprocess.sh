#!/bin/bash

name="chrome_debian"
inpdir="data/$name"
outpdir="out/$name"

mkdir -p "$outpdir"

export PYTHONPATH="$PWD:$PYTHONPATH"
python data_processing/preprocess.py --project "$name" --input "$inpdir/" --output "$outpdir/" || exit 1;

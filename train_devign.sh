#!/bin/bash

name="chrome_debian$1"
outpdir="out/$name"

mkdir -p "$outpdir" "$outpdir/models"

python Devign/main.py --model_type devign --input_dir "$outpdir/ggnn_input/" --model_dir "$outpdir/models/" || exit 1

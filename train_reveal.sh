#!/bin/bash

name="chrome_debian$1"
outpdir="out/$name"

mkdir -p "$outpdir" "$outpdir/models"

python Devign/main.py --model_type ggnn --input_dir "$outpdir/ggnn_input/" --model_dir "$outpdir/models/" || exit 1;

python Devign/run_model.py --model_dir "$outpdir/models/" --dataset "$outpdir/ggnn_input/processed.bin" --output_dir "$outpdir/after_ggnn/" --name "$name" || exit 1;

python Vuld_SySe/representation_learning/api_test.py --dataset "$outpdir/after_ggnn/" --features ggnn;

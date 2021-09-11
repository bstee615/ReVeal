#!/bin/bash

input_dir="$1"

python Devign/main.py --model_type ggnn --input_dir "$input_dir/ggnn_input/" --model_dir "$input_dir/models/" --seed 0 || exit 1;

python Devign/run_model.py --dataset "$input_dir/ggnn_input/processed.bin" --model_dir "$input_dir/models/" --output_dir "$input_dir/after_ggnn/" || exit 1;

python Vuld_SySe/representation_learning/api_test.py --dataset "$input_dir/after_ggnn/" --model_dir "$input_dir/models/" --seed 0 --features ggnn;

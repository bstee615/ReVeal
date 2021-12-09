#!/bin/bash

set -x

echo "$@"

input_dir="$1"
ray_flag="$2"

#python Devign/main.py --model_type ggnn --input_dir "$input_dir/" --model_dir "$input_dir/models/" --seed 0 || exit 1;

#python Devign/run_model.py --dataset "$input_dir/ggnn_input/processed.bin" --model_dir "$input_dir/models/" --output_dir "$input_dir/after_ggnn/" || exit 1;

#python Vuld_SySe/representation_learning/api_test.py --dataset "$input_dir/after_ggnn/" --model_dir "$input_dir/models/" --seed 0 --features ggnn;

#python Devign/main.py --model_type ggnn --input_dir "$input_dir/" --seed 0 --save_after || exit 1;

python Vuld_SySe/representation_learning/api_test.py --dataset "$input_dir/after_ggnn/" --model_dir "$input_dir/models/" --seed 0 --features ggnn;

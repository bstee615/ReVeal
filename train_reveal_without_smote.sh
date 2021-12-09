#!/bin/bash

echo "$@"

input_dir="$1"

python Devign/main.py --model_type ggnn --input_dir "$input_dir/" --seed 0 --save_after || exit 1;

python Vuld_SySe/representation_learning/api_test.py --dataset "$input_dir/after_ggnn/" --model_dir "$input_dir/models/" --seed 0 --features ggnn --no_balance || exit 1;

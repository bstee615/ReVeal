#!/bin/bash

input_dir="$1"

python Devign/main.py --model_type devign --input_dir "$input_dir/ggnn_input/" --model_dir "$input_dir/models/" --seed 0 || exit 1

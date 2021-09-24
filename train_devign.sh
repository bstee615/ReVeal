#!/bin/bash

echo "$@"

input_dir="$1"

python Devign/main.py --model_type devign --input_dir "$input_dir/" --seed 0 || exit 1

#!/bin/bash

set -x

echo "$@"

input_dir="$1"
ray_flag="$2"

python -u Devign/main.py --model_type devign --input_dir "$input_dir/" --seed 0 || exit 1

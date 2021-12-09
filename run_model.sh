#!/bin/bash

set -x

if [ $# -lt 2 ] || [ -z "$1" ] || [ -z "$2" ]
then
  echo "Usage: $0 <model> <directory_to_analyze>" && exit
fi

train_script="train_$1.sh"
if [ ! -f "$train_script" ]
then
  echo "Does not exist: $train_script" && exit
fi

input_dir="$2"
if [ ! -d "$input_dir" ]
then
  echo "Does not exist: $input_dir" && exit
fi

ray_flag="$3"

srun -J "$train_script" --output "$input_dir/sbatch-%j.out" batch.sh bash "$train_script" "$input_dir" "$ray_flag"

#name="$1-$(echo "$input_dir" | rev | cut -d'/' -f1 | rev)"
#echo "Name: $name"

#srun -J "$name" \
#  --time=06:00:00 \
#  --nodes=1 --pty \
#  bash batch.sh bash "$train_script" "$input_dir"
#  --gres=gpu:1 --partition=gpu --exclude=amp-1 \
#  bash batch.sh bash "$train_script" "$input_dir"

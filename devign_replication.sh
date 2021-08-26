#!/bin/bash

# ./full_run.sh -i input_dir (parsed cpgs) -o output_dir -n name of folder

while getopts i:o:n: flag
do
    case "${flag}" in
        i) inpdir=${OPTARG};;
        o) outpdir=${OPTARG};;
        n) name=${OPTARG};;
    esac
done

mkdir -p "$outpdir/data/full_experiment_real_data/$name";
mkdir -p "$outpdir/data/full_experiment_real_data_processed";
mkdir -p "$outpdir/data/models/$name";
mkdir -p "$outpdir/data/ggnn_input";
mkdir -p "$outpdir/data/after_ggnn/$name";

python Devign/main.py --model_type devign --dataset "$name" --input_dir "data/ggnn_input/$name/" --model "data/models/$name/DevignModel.bin" --feature_size 169 --patience 100 || exit 1;


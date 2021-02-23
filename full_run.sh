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

mkdir -p "$outpdir/$name/data/full_experiment_real_data";
mkdir -p "$outpdir/$name/data/full_experiment_real_data_processed";
mkdir -p "$outpdir/$name/model";
mkdir -p "$outpdir/$name/ggnn_input";
mkdir -p "$outpdir/$name/after_ggnn";

python data_processing/create_ggnn_input.py --input "$inpdir" --output "$outpdir/$name/data/";

python data_processing/extract_slices.py #arguments need to be added to script (currently hard coded)

python data_processing/create_ggnn_data.py --input "$outpdir/$name/data/" --project "$name" --csv "$inpdir/parsed/" --src "$inpdir/raw_code/" --wv "data/chrome_debian/raw_code_deb_chro.100" --output "$outpdir/$name/data/full_experiment_real_data/$name.json"

python data_processing/full_data_prep_script.py #arguments and paths all hard coded needs changing

python data_processing/split_data.py --input "$outpdir/$name/data/full_experiment_real_data_processed/$name-full_graph.json" --output "$outpdir/$name/ggnn_input" --percent 50 --name "$name"

python ../Devign/main.py --model_type ggnn --dataset "$name" --input_dir "$outpdir/$name/ggnn_input/" --feature_size 169

python ../Devign/run_model.py --model "models/$name/GGNNSumModel-model.bin" --dataset "$outpdir/$name/ggnn_input/processed.bin" --output_dir "$outpdir/$name/after_ggnn" --name "$name"

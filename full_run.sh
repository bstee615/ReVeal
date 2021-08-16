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
mkdir -p "$outpdir/data/model";
mkdir -p "$outpdir/data/ggnn_input";
mkdir -p "$outpdir/data/after_ggnn/$name";

python data_processing/create_ggnn_input.py --project "$name" --input "$inpdir" --output "$outpdir/data/" || exit 1;

python data_processing/extract_slices.py --project "$name" --input "$inpdir" --text_in "$outpdir/data/" --output "$outpdir/data/" || exit 1;

python data_processing/create_ggnn_data.py --input "$outpdir/data/" --project "$name" --csv "$inpdir/parsed/" --src "$inpdir/raw_code/" --wv "data/chrome_debian/raw_code_deb_chro.100" --output "$outpdir/data/full_experiment_real_data/$name/$name.json" || exit 1;

python data_processing/full_data_prep_script.py --project "$name" --input "$inpdir" --base "$outpdir/data/full_experiment_real_data/" --output "$outpdir/data/full_experiment_real_data_processed" || exit 1;

python data_processing/split_data.py --input "$outpdir/data/full_experiment_real_data_processed/$name-full_graph.json" --output "$outpdir/data/ggnn_input" --name "$name" || exit 1;

python Devign/main.py --model_type ggnn --dataset "$name" --input_dir "$outpdir/data/ggnn_input/$name/" --model "$outpdir/data/models/$name/GGNNSumModel.bin" --feature_size 169 || exit 1;

python Devign/run_model.py --model "$outpdir/data/models/$name/GGNNSumModel.bin" --dataset "$outpdir/data/ggnn_input/$name/processed.bin" --output_dir "$outpdir/data/after_ggnn/" --name "$name" || exit 1;

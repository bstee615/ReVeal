#!/bin/bash

# ./full_run.sh -i input_dir (parsed cpgs) -o output_dir -n name of folder

while getopts i:o:n:t flag
do
    case "${flag}" in
        i) inpdir=${OPTARG};;
        o) outpdir=${OPTARG};;
        n) name=${OPTARG};;
        t) test=1;;
    esac
done

if [ ! -z "$test" ]
then
    echo inpdir=$inpdir\; outpdir=$outpdir\; name=$name
    # cat "$0"
    exit 0
fi

mkdir -p "$outpdir/data/full_experiment_real_data/$name";
mkdir -p "$outpdir/data/full_experiment_real_data_processed";
mkdir -p "$outpdir/data/model";
mkdir -p "$outpdir/data/ggnn_input";
mkdir -p "$outpdir/data/after_ggnn/$name";

python Devign/initialize_data.py --dataset "$name" --input_dir "$outpdir/data/ggnn_input/$name/" || exit 1;

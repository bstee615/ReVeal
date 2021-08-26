#!/bin/bash

count="$(realpath $(dirname $0))/count.py"

function run_for_dataset()
{
    echo "$1":
    (cd "$1"; python $count)
}

run_for_dataset data/after_ggnn/chrome_debian/imbalance/v3
run_for_dataset out/data/after_ggnn/chrome_debian
run_for_dataset out/data/ggnn_input/chrome_debian

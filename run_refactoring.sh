#!/bin/bash
source source.sh
source cfactor/load.sh
python data_processing/refactor_reveal.py -i "data/devign" -o "out/refactored_devign_no_new_name" --nproc 15 --no-new-names
python data_processing/add_refactored_code.py --input_dir "out/refactored_devign_no_new_name"
python data_processing/preprocess.py -i "data/devign" "out/refactored_devign_no_new_name" -o "data/devign_no_new_name"
python Devign/main.py --model_type devign --input_dir "data/devign_no_new_name/ggnn_input/" --model_dir "data/devign_no_new_name/models/" --seed 0

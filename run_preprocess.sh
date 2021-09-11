#!/bin/bash
source source.sh
python data_processing/preprocess.py --project chrome_debian --input data/chrome_debian data/chrome_debian_refactored2 --output basic_plus_refactored_cross_validated --with_augmented
python data_processing/preprocess.py --project chrome_debian --input data/chrome_debian --output basic

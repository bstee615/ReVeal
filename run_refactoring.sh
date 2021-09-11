#!/bin/bash
source source.sh
source cfactor/load.sh
python data_processing/refactor_reveal.py -i data/chrome_debian
python data_processing/add_refactored_code.py --output_dir data/chrome_debian_refactored2 2>&1 | tee add_refactored_code.py.log

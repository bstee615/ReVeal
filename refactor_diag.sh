#!/bin/bash

dir="$1"
python -u data_processing/refactor_reveal.py --mode diag -i "$dir" -o "$dir/refactored_pickle"

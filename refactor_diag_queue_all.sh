#!/bin/bash
set -x

sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output_thresholds/devign_threshold_0.25/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output_thresholds/devign_threshold_0.75/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output_thresholds/devign_threshold_1.0/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output_thresholds/devign_no-new-names_threshold_0.25/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output_thresholds/devign_no-new-names_threshold_1.0/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output/devign_threshold_0.5_no-new-names/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/devign         -o output/devign_no-new-names_threshold_0.75/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_threshold_0.25/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_threshold_0.5/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_threshold_0.75/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_threshold_1.0/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_no-new-names_threshold_0.25/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_no-new-names_threshold_0.5/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_no-new-names_threshold_0.75/refactored_pickle
sbatch cpu_batch.sh python -u data_processing/refactor_reveal.py --mode diag -i data/chrome_debian  -o output_thresholds/chrome_debian_no-new-names_threshold_1.0/refactored_pickle

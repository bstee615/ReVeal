#!/bin/bash
#SBATCH -t8:00:00
#SBATCH -n 16
#SBATCH -p interactive

module load miniconda3
source activate $PWD/../env
bash data_prep.sh -i "$PWD/data_refactored/chrome_debian/" -o "$PWD/out_refactored/" -n chrome_debian

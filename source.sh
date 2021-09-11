#!/bin/bash
# initialize environment for slurm node to run refactoring or preprocessing
module load gcc
module load curl
module load libarchive
module load openssl
module load miniconda3
source activate ReVeal
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/cfactor/srcml/lib"
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/cfactor"


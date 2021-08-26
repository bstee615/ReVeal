#!/bin/bash

#SBATCH --time=2-00:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 16 processor core(s) per node 
#SBATCH --exclude=amp-1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --mail-user=benjis@iastate.edu   # email address
#SBATCH --mail-type=FAIL
#SBATCH --output="sbatch-%j.out" # job standard output file (%j replaced by job id)

envdir="$PWD/../env"
echo Environment: $envdir Current directory: $PWD
module load miniconda3
source activate "$envdir"
bash -x devign.sh -i "$PWD/data/chrome_debian/" -o "$PWD/out/" -n chrome_debian


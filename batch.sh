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

source load_all.sh

$@

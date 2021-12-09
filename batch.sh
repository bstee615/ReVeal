#!/bin/bash

#SBATCH --time=2-00:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --exclude=amp-1,amp-2,amp-3
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --mail-user=benjis@iastate.edu   # email address
#SBATCH --mail-type=FAIL,END
#SBATCH --output="sbatch-%j.out" # job standard output file (%j replaced by job id)

source load_all.sh

$@

#!/bin/bash
#SBATCH -t8:00:00
#SBATCH -n 16
#SBATCH -p interactive
#SBATCH --exclude=legion-[1-8]
#SBATCH --mem 32GB
#SBATCH --output="sbatch-%j.out" # job standard output file (%j replaced by job id)

source load_all.sh
echo "$@"
$@

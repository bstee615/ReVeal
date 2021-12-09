#!/bin/bash
#SBATCH -t8:00:00
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=legion-[1-8]
#SBATCH --mem 32GB
#SBATCH -p whatever
#SBATCH --mail-user=benjis@iastate.edu   # email address
#SBATCH --mail-type=FAIL,END
#SBATCH --output="sbatch-%j.out" # job standard output file (%j replaced by job id)

source load_all.sh
echo "$@"
$@

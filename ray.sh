#!/bin/bash
#SBATCH --job-name=tune
##SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=benjis@iastate.edu
##SBATCH --mail-type=FAIL,END
#SBATCH --output="sbatch-%j.out"

source load_all.sh

set -x
#num_gpus="1"
#num_nodes=""

# Getting the node names
#nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
#nodes_array=($nodes)

#head_node=${nodes_array[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
#if [[ "$head_node_ip" == *" "* ]]; then
#IFS=' ' read -ra ADDR <<<"$head_node_ip"
#if [[ ${#ADDR[0]} -gt 16 ]]; then
#  head_node_ip=${ADDR[1]}
#else
#  head_node_ip=${ADDR[0]}
#fi
#echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
#fi

#port=6379
#ip_head=$head_node_ip:$port
#export ip_head
#echo "IP Head: $ip_head"

#echo "Starting HEAD at $head_node"
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    ray start --head --node-ip-address="localhost" --port=$port \
#    --num-cpus "1" --num-gpus "0" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
#sleep 10

# number of nodes other than the head node
#for ((i = 1; i <= "$num_nodes"; i++)); do
#    node_i=${nodes_array[$i]}
#    echo "Starting WORKER $i at $node_i"
#    srun --nodes=1 --ntasks=1 -w "$node_i" \
#        ray start --address "localhost" \
#        --num-cpus "4" --num-gpus "$num_gpus" --block &
#    sleep 5
#done

# ray/doc/source/cluster/examples/simple-trainer.py
#python -u simple_trainer.py "$SLURM_CPUS_PER_TASK"
#python -u data_processing/tune.py --address "$ip_head"
#python -u data_processing/tune.py --test --cpus_per_task 4 --gpus_per_task 0 --num_cpus 9 --num_gpus 0
python -u data_processing/tune.py --cpus_per_task 4 --gpus_per_task 1 --num_cpus 5 --num_gpus 1

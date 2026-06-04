#!/bin/bash
#SBATCH -A gen006
#SBATCH -J check_jax
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
##SBATCH --cpus-per-task=56
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err

module load rocm/7.1.1
module load rccl-net-plugin
# git switch umain

source /lustre/orion/gen006/world-shared/booshan/NuLattice/.venv/bin/activate

ulimit -c 0 # disable core dumps

# export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_dump_to=.hlo_dump/$SLURM_JOB_NAME --xla_dump_hlo_as_text"
# export XLA_FLAGS="--xla_gpu_autotune_level=0 --xlo_dump_hlo_pass_re=shardy"
# export XLA_PYTHON_CLIENT_ALLOCATOR=default
 
# export NCCL_ALGO=Tree
# export NCCL_CROSS_NIC=1
# export NCCL_BUFFSIZE=16777216
# export NCCL_TIMEOUT=1800
# export NCCL_DEBUG=INFO

# export RCCL_TIMEOUT=1800
# export RCCL_DEBUG=INFO

# export JAX_USE_SHARDY_PARTITIONER=0
# export JAX_COMPILATION_CACHE_DIR="./.jax_cache"
# export JAX_TRACEBACK_FILTERING=off
# export JAX_ENABLE_X64=True
# export ROCR_VISIBLE_DEVICES=0,1,2,3
# OUTPUT_FILE="data/node_scaling_results_jax.csv"

# L=6
# REFERENCE="o16"

# echo "Node $(hostname) is handling L=$L"

start=$SECONDS

srun python check_jax.py

elapsed=$(( SECONDS - start ))

printf "took %02d:%02d:%02d\n" "$((elapsed/3600))" "$((elapsed%3600/60))" "$((elapsed%60))"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,NodeList,Elapsed,MaxRSS --parsable2 --delimiter="," | grep $REFERENCE >> $OUTPUT_FILE


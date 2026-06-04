#!/bin/bash
#SBATCH -A gen006
#SBATCH -J o16gs_30_hf_jcpu
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

# --- Environment Setup ---
# module load rocm/6.2.4
source /lustre/orion/gen006/world-shared/booshan/NuLattice/.venv/bin/activate

export JAX_ENABLE_X64=True
export JAX_TRACEBACK_FILTERING=off

# Prevent JAX from pre-allocating 75% of VRAM immediately. 
# This gives the ShardingManager more room to move sparse buffers.
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=default
# export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS="--xla_gpu_autotune_level=0"

# Frontier Specific: High-speed interconnect tuning for sharding
# export NCCL_NET_GDR_LEVEL=3
# export HSA_XNACK=1
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5
OUTPUT_FILE="hf_slurm_data.csv"
REFERENCE="o16"
start=$SECONDS

srun uv run python Example_HF.py \
  --backend jax \
  --reference $REFERENCE \
  --L 30 \
  --a_lat 2.0 --vT1 -8.0 --vS1 -8.0 --cE 5.5 \
  # --use_davidson \
  # --backend jax --shard

elapsed=$(( SECONDS - start ))
printf "took %02d:%02d:%02d\n" "$((elapsed/3600))" "$((elapsed%3600/60))" "$((elapsed%60))"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,NodeList,Elapsed,MaxRSS --parsable2 --delimiter="," | grep $REFERENCE >> $OUTPUT_FILE

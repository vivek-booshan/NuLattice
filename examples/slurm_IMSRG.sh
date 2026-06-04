#!/bin/bash
#SBATCH -A gen006
#SBATCH -J FCI_4
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1

module load rocm/6.2.4
source /lustre/orion/gen006/world-shared/booshan/NuLattice/.venv/bin/activate
export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_gpu_autotune_level=0"

start=$SECONDS
srun uv run python Example_IMSRG.py --backend jax --L 3 --element he3
elapsed=$(( SECONDS - start ))

printf "took %02d:%02d:%02d\n" "$((elapsed/3600))" "$((elapsed%3600/60))" "$((elapsed%60))"
# sacct -j $SLURM_JOB_ID --format=JobID,JobName,NodeList,Elapsed,MaxRSS --parsable2 --delimiter="," | grep $REFERENCE >> $OUTPUT_FILE



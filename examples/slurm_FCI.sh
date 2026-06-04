#!/bin/bash
#SBATCH -A gen006
#SBATCH -J FCI_4
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1

start=$SECONDS

srun uv run python Example_FCI.py

elapsed=$(( SECONDS - start ))
printf "took %02d:%02d:%02d\n" "$((elapsed/3600))" "$((elapsed%3600/60))" "$((elapsed%60))"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,NodeList,Elapsed,MaxRSS --parsable2 --delimiter="," | grep $REFERENCE >> $OUTPUT_FILE


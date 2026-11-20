#!/bin/bash
#SBATCH --job-name=covar_windows
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-11-20: every 5-10 nm window, model (same up to a scalar per sky) against the PFS sky (pfs_covar/test_windows.py)
module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARMS=(b r n m)
python -W ignore pfs_covar/test_windows.py ${ARMS[$SLURM_ARRAY_TASK_ID]}

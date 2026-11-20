#!/bin/bash
#SBATCH --job-name=ragnar_fix4
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-10-07: line lists for the pipeline team. As _fix3 (hot population, f8/f9, per-arm sigma and wing), with LSF_err = 0.05. Centroider lists only.

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARMS=(b r n)
/usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm ${ARMS[$SLURM_ARRAY_TASK_ID]} -cent 1 -tag _fix4

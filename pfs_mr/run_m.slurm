#!/bin/bash
#SBATCH --job-name=ragnar_m
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-11-16: the PFS medium-resolution (m arm) list, centroider, fix4 grid. sigma, wing and range are the m defaults in run_RAGNAR.py
# (sigma 0.0637 chosen over 0.0613 against the data, pfs_mr/test_linelist.py; the wing does not change the centroids)

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm m -cent 1 -tag ${TAG:-_fix4wide}

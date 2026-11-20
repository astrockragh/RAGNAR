#!/bin/bash
#SBATCH --job-name=covar_model
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-11-20: the simulated spectra of the cent=0 fix4 runs through the same line measurement as the PFS data (pfs_covar/measure.py)
module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -W ignore pfs_covar/measure.py model

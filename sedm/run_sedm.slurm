#!/bin/bash
#SBATCH --job-name=ragnar_sedm
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-11-16: SEDM sky-line questions. R = 100 with and without the centroider, and R = 75 / 150 to bracket the prism resolution.
# Flux calibration is done on the runs without the centroider (cent = 0); the centroider runs are for the wavelength calibration.

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# TAG: '' was the first (Palomar-like) grid; _wst is the Wendelstein-like default grid of run_sedm.py
TAG=${TAG:-_wst}
RS=(100 100 75 150 75 150)
CS=(1 0 1 1 0 0)
/usr/bin/time -v python -W ignore::FutureWarning sedm/run_sedm.py -R ${RS[$SLURM_ARRAY_TASK_ID]} -cent ${CS[$SLURM_ARRAY_TASK_ID]} -seed 0 -tag ${TAG}

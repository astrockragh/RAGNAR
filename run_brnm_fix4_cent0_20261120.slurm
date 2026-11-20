#!/bin/bash
#SBATCH --job-name=ragnar_covar
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-11-20: no-centroider (cent=0) lists of all four PFS arms with the fix4 settings, for relative flux calibration.
# Same grid and seed 0 in every arm, so the arms simulate the same skies and their lines can be grouped together
# (covariation.combine_covariation on the *_covar.npz files).

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARMS=(b r n m)
/usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm ${ARMS[$SLURM_ARRAY_TASK_ID]} -cent 0 -tag _fix4

#!/bin/bash
#SBATCH --job-name=ragnar_fix3
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-10-07 rerun: hot population from PFS + Noll+ 2020 (R_HOT, T_HOT), r_hot scales 0.75/1/1.33, f9 = 1.35, on top of _fix2. Outputs tagged _fix3.
# with the same seed so the cent=0/1 pairs share identical simulated spectra.

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARMS=(b r n b r n)
CENTS=(0 0 0 1 1 1)
V=/scratch/gpfs/cj1223/ragnar_validate

case $SLURM_ARRAY_TASK_ID in
  [0-5]) /usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm ${ARMS[$SLURM_ARRAY_TASK_ID]} -cent ${CENTS[$SLURM_ARRAY_TASK_ID]} -tag _fix3 ;;
esac

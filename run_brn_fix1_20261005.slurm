#!/bin/bash
#SBATCH --job-name=ragnar_fix1
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-10-05 rerun after the physics and line-matching fixes (see validation/TIER2_CHECKLIST.md). Outputs are tagged _fix1.
# with the same seed so the cent=0/1 pairs share identical simulated spectra.

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARMS=(b r n b r n)
CENTS=(0 0 0 1 1 1)
V=/scratch/gpfs/cj1223/ragnar_validate

case $SLURM_ARRAY_TASK_ID in
  [0-5]) /usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm ${ARMS[$SLURM_ARRAY_TASK_ID]} -cent ${CENTS[$SLURM_ARRAY_TASK_ID]} -tag _fix1 ;;
esac

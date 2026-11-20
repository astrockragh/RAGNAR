#!/bin/bash
#SBATCH --job-name=ragnar_brn
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-10-05: optimized RAGNAR. Tasks 0-5 make the b/r/n line lists without (cent=0) and with (cent=1) the centroider,
# with the same seed so the cent=0/1 pairs share identical simulated spectra.
# Tasks 6-7 rerun the full r arm with the original code and the new code at the same seed, to check they agree exactly.

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARMS=(b r n b r n)
CENTS=(0 0 0 1 1 1)
V=/scratch/gpfs/cj1223/ragnar_validate

case $SLURM_ARRAY_TASK_ID in
  [0-5]) /usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm ${ARMS[$SLURM_ARRAY_TASK_ID]} -cent ${CENTS[$SLURM_ARRAY_TASK_ID]} ;;
  6) cd $V && /usr/bin/time -v python -W ignore cmp_run.py $V/orig 600 1000 0 $V/orig_r_full.pkl 0.108 ;;
  7) cd $V && /usr/bin/time -v python -W ignore cmp_run.py /home/cj1223/OHspec/make_linelists/RAGNAR_new 600 1000 0 $V/new_r_full.pkl 0.108 ;;
esac

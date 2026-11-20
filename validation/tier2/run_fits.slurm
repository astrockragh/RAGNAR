#!/bin/bash
#SBATCH --job-name=sky_fits
#SBATCH --account=astro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new/validation/tier2
# 4 visits x 3 arms; extra arguments (e.g. --band, --free-wing) are passed through: sbatch run_fits.slurm --free-wing
module purge; module load anaconda3/2025.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
VISITS=(129688 129689 129806 129807)
ARMS=(b r n)
python -W ignore fit_all.py ${VISITS[$((SLURM_ARRAY_TASK_ID / 3))]} ${ARMS[$((SLURM_ARRAY_TASK_ID % 3))]} "$@"

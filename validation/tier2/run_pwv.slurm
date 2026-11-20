#!/bin/bash
#SBATCH --job-name=pwv_scan
#SBATCH --account=astro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new/validation/tier2
module purge; module load anaconda3/2025.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
VISITS=(129688 129689 129806 129807)
python -W ignore pwv_scan.py ${VISITS[$SLURM_ARRAY_TASK_ID]}

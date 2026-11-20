#!/bin/bash
#SBATCH --job-name=ragnar_mtest
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new

# 2026-11-16: PFS medium resolution (m arm). Window-only (712-780 nm) lists to choose sigma and the wing against the data in pfs_mr/test_linelist.py:
# A = core/FWHM sigma with the fitted wing, B = the +-0.9 nm Gaussian-fit sigma with the same wing, C = core sigma without a wing

module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SIG=(0.0637 0.0613 0.0637)
RSC=(0.006 0.006 0.0)
TAG=(_mtestA _mtestB _mtestC)
i=$SLURM_ARRAY_TASK_ID
/usr/bin/time -v python -W ignore::FutureWarning run_RAGNAR.py -ls 3 -seed 0 -arm m -cent 1 -sig ${SIG[$i]} -rscat ${RSC[$i]} -wf 8.0 -wmin 712 -wmax 780 -spec_dir '' -tag ${TAG[$i]}

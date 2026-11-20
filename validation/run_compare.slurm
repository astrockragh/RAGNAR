#!/bin/bash
#SBATCH --job-name=compare_lists
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new/validation
module purge; module load anaconda3/2025.12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
declare -A SIG=( [b]=0.088 [r]=0.112 [n]=0.105 )
for a in b r n; do
  python -W ignore compare_linelists.py $a --fibres 40 --sim-spectra $(ls /tigress/cj1223/spectra/sweep_0_${SIG[$a]}_${a}_fix3_*.pkl | head -1) > logs/compare_$a.log 2>&1 &
done; wait

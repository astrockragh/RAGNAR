#!/bin/bash
#SBATCH --job-name=lsf_pfsarm
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new/validation/tier2
# the three arms in parallel on one node (a whole node starts sooner in the pu partition)
module purge; module load anaconda3/2025.12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -W ignore lsf_pfsarm.py b --fix-wing 0.0125 3.4 --contam 0.05 > logs/lsf_b_fixwing.log 2>&1 &
python -W ignore lsf_pfsarm.py r --fix-wing 0.006944 3.4 > logs/lsf_r_fixwing.log 2>&1 &
python -W ignore lsf_pfsarm.py n --fix-wing 0.006944 3.4 > logs/lsf_n_fixwing.log 2>&1 &
wait

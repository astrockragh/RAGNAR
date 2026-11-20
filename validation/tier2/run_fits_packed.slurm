#!/bin/bash
#SBATCH --job-name=sky_fits
#SBATCH --account=astro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/home/cj1223/OHspec/make_linelists/RAGNAR_new/validation/tier2
# all 36 fits (4 visits x 3 arms x {per-v', per-band, free wing}) in one job, 12 at a time with 4 threads each (a whole node goes to the pu partition and starts sooner than a partial one in serial):
# the QOS limits how many jobs can be queued, so they are packed into one
module purge; module load anaconda3/2025.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# fit_all.py ignores --vup (the default mode); it's there so no line ends in a blank, which xargs -L would join with the next line
OPTS=${OPTS:-"--vup --band --free-wing"}
for opt in $OPTS; do for v in 129688 129689 129806 129807; do for a in b r n; do echo "$v $a $opt"; done; done; done \
  | xargs -P 12 -L 1 sh -c 'python -W ignore fit_all.py $0 $1 $2 > logs/fit_$0_$1${2}.log 2>&1'
echo done

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_sunburst
#
# Partition:
#SBATCH --partition=savio2
#
# Wall clock limit:
#SBATCH --time=05:00:00
#
## Command(s) to run:
module unload python # make sure python/2 not loaded
module load python

ka0=0.6
mut=20
mlow=5
mhigh=8
Aacc=0.08
outstr="analyze"
outstr+=$i
outstr+=".out"

python calc_mags.py $ka0 $mut $mlow $mhigh $Aacc $i > $outstr

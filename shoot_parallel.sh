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
#SBATCH --time=10:00:00
#
## Command(s) to run:
module unload python # make sure python/2 not loaded
module load python
    
ka0=0.6
mut=10
mlow=5
mhigh=8
no_runs=10
outstr="test"
outstr+=$i
outstr+=".out"
    
python rayshoot.py $ka0 $mut $mlow $mhigh $no_runs $i > $outstr

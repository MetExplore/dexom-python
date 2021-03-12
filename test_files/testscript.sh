#!/bin/bash
#SBATCH -J test
#SBATCH -p workq
#SBATCH -o testoutput.out
#SBATCH -e testerror.out
#SBATCH -t 00:01:00
#SBATCH --mem=2G
#SBATCH --mail-type=BEGIN,END,FAIL
#Purge any previous modules
module purge

module load system/Python-3.7.4

# My command lines I want to run on the cluster
python /home/mstingl/work/dexom_py/test_files/maintest.py
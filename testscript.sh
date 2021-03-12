#!/bin/bash
#SBATCH -J test
#SBATCH -p workq
#SBATCH -o testoutput.out
#SBATCH -e testerror.out
#SBATCH -t 00:01:00
#SBATCH --mem=2G
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module purge
module load system/Python-3.7.4

source env/bin/activate
export PYTHONPATH=${PYTHONPATH}:/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux

python test_files/maintest.py
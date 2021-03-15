#!/bin/bash
#SBATCH -J main_script
#SBATCH -p workq
#SBATCH -o mainout.out
#SBATCH -e mainerr.out
#SBATCH --mail-type=ALL
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 00:50:00

cd $SLURM_SUBMIT_DIR

module purge

module load system/Python-3.7.4

source env/bin/activate
export PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux"
echo $PYTHONPATH

python src/main.py

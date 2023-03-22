#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -J submit
#SBATCH -o submit_out.out
#SBATCH -e submit_err.out

cd $SLURM_SUBMIT_DIR
module purge
module load system/Python-3.7.4
source env/bin/activate
export PYTHONPATH=${PYTHONPATH}:"/home/mstingl/save/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux"

snakemake --forceall --dag | grep -v "Restricted*" | grep -v "No*" | dot -Tpdf > dag.pdf
snakemake --cluster "python3 submit.py {dependencies}" --immediate-submit --notemp -j 500


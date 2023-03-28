#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -J submit
#SBATCH -o submit_out.out
#SBATCH -e submit_err.out

cd $SLURM_SUBMIT_DIR
module purge
module load system/Python-3.7.4
source env/bin/activate
export PYTHONPATH=${PYTHONPATH}:"path/to/CPLEX_solver"

snakemake -s dexom_python/cluster_utils/Snakefile --cluster "python3 dexom_python/cluster_utils/submit.py {dependencies}" --immediate-submit --notemp -j 500


#!/bin/bash
#SBATCH -p workq
#SBATCH --mail-type=ALL
#SBATCH --mem=64G
#SBATCH -c 24
#SBATCH -t 01:00:00
#SBATCH -J gpr_rules
#SBATCH -o gpr_rules.out
#SBATCH -e gpr_rules.err

module purge
module load system/Python-3.7.4
source env/bin/activate
export PYTHONPATH=${PYTHONPATH}:"/path/to/solver/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux"

python dexom_python/gpr_rules.py -m example_data/recon2v2_corrected.json -g example_data/pval_0-01_geneweights.csv -o example_data/pval_0-01_reactionweights

# this file can be used as an example for executing dexom_python functions on a slurm cluster
# Note that the "path/to/solver" must be replaced by the folder where the CPLEX solver is actually installed
# The example given here uses the gpr_rules.py script to convert gene expression values into reaction weights, see documentation for more possible functions
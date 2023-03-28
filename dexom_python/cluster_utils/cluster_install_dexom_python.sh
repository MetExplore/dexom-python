#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -J cluster_install
#SBATCH -o cluster_install_out.out
#SBATCH -e cluster_install_err.out

cd $SLURM_SUBMIT_DIR
git clone https://forgemia.inra.fr/metexplore/cbm/dexom-python.git dexompy
cd dexompy

module purge
module load system/Python-3.7.4

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install poetry
poetry install
pip install snakemake

echo "installation complete"

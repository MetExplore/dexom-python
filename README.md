# DEXOM in python

<a href = "https://github.com/MetExplore/dexom-python/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/maximiliansti/dexom_python"></a>
<a href="https://pypi.org/project/dexom-python/"><img alt = "PyPI Package" src = "https://img.shields.io/pypi/v/dexom-python"/></a>  

This is a python implementation of DEXOM (Diversity-based enumeration of optimal context-specific metabolic networks)  
The original project, which was developped in MATLAB, can be found here: https://github.com/MetExplore/dexom  
Parts of the imat code were taken from the driven package for data-driven constraint-based analysis: https://github.com/opencobra/driven

API documentation is available here: https://dexom-python.readthedocs.io/en/stable/

The package can be installed using pip: `pip install dexom-python`

You can also clone the git repository with `git clone https://github.com/MetExplore/dexom-python` and then install dependencies with `python setup.py install`

To view changes between versions, see [changelog](docs/changelog.rst)

## Requirements
- Python 3.7 - 3.9
- CPLEX 12.10 - 22.10

### Installing CPLEX

[Free license (Trial version)](https://www.ibm.com/analytics/cplex-optimizer): this version is limited to 1000 variables and 1000 constraints, and is therefore not useable on larger models

[Academic license](https://www.ibm.com/academic/technology/data-science): for this, you must sign up using an academic email address.
 - after logging in, you can access the download for "ILOG CPLEX Optimization Studio"
 - download version 12.10 or higher of the appropriate installer for your operating system
 - install the solver 

You must then update the PYTHONPATH environment variable by adding the directory containing the `setup.py` file appropriate for your OS and python version   
Alternatively, run `python "C:\Program Files\IBM\ILOG\CPLEX_Studio1210\python\setup.py" install` and/or `pip install cplex==12.10`

## Functions

These are the different functions which are available for context-specific metabolic subnetwork extraction

### apply_gpr
The `gpr_rules.py` script can be used to transform gene expression data into reaction weights, for a limited selection of models.  
It uses the gene identifiers and gene-protein-reaction rules present in the model to connect the genes and reactions.  
By default, continuous gene expression values/weights will be transformed into continuous reaction weights.  
Using the `--convert` flag will instead create semi-quantitative reaction weights with values in {-1, 0, 1}. By default, the proportion of these three weights will be {25%, 50%, 25%}.

### iMAT
`imat.py` contains a modified version of the iMAT algorithm as defined by [(Shlomi et al. 2008)](https://pubmed.ncbi.nlm.nih.gov/18711341/).  
The main inputs of this algorithm are a model file, which must be supplied in a cobrapy-compatible format (SBML, JSON or MAT), and a reaction_weight file in which each reaction is attributed a score.  
These reaction weights must be determined prior to launching imat, for example with GPR rules present in the metabolic model.  

The remaining inputs of imat are:
- `epsilon`: the activation threshold of reactions with weight > 0
- `threshold`: the activation threshold for unweighted reactions
- `full`: a bool parameter for switching between the partial & full-DEXOM implementation

In addition, the following solver parameters have been made available through the solver API:
- `timelimit`: the maximum amount of time allowed for solver optimization (in seconds)
- `feasibility`: the solver feasbility tolerance
- `mipgaptol`: the solver MIP gap tolerance
note: the feasibility determines the solver's capacity to return correct results. 
In particular, it is necessary that `epsilon` > `threshold` > `ub*feasibility` (where `ub` is the maximal upper bound for reaction flux in the model)

By default, imat uses the `create_new_partial_variables` function. In this version, binary flux indicator variables are created for each reaction with a non-zero weight.  
In the full-DEXOM implementation, binary flux indicator variables are created for every reaction in the model. This does not change the result of the imat function, but can be used for the enumeration methods below.

### enum_functions

Four methods for enumerating context-specific networks are available:
- `rxn_enum_functions.py` contains reaction-enumeration (function name: `rxn_enum`)
- `icut_functions.py` contains integer-cut (function name: `icut`)
- `maxdist_functions.py` contains distance-maximization (function name: `maxdistm`)
- `diversity_enum_functions.py` contains diversity-enumeration  (function name: `diversity_enum`)

An explanation of these methods can be found in [(Rodriguez-Mier et al. 2021)](https://doi.org/10.1371/journal.pcbi.1008730).  
Each of these methods can be used on its own. The same model and reaction_weights inputs must be provided as for the imat function.

Additional parameters for all 4 methods are:
- `prev_sol`: an imat solution used as a starting point (if none is provided, a new one will be computed)  
- `obj_tol`: the relative tolerance on the imat objective value for the optimality of the solutions  
icut, maxdist, and diversity-enum also have two additional parameters:
- `maxiter`: the maximum number of iterations to run
- `full`: set to True to use the full-DEXOM implementation  
As previously explained, the full-DEXOM implementation defines binary indicator variables for all reactions in the model. Although only the reactions with non-zero weights have an impact on the imat objective function, the distance maximization function which is used in maxdist and diversity-enum can utilize the binary indicators for all reactions. This increases the distance between the solutions and their diversity, but requires significantly more computation time.  
maxdist and div-enum also have one additional parameter:  
- `icut`: if True, an icut constraint will be applied to prevent duplicate solutions

## Parallelized DEXOM

The DEXOM algorithm is a combination of several network enumeration methods.  
`write_cluster_scripts.py` contains functions which are used for creating a parallelization of DEXOM on a slurm computation cluster.
The default function is `write_batch_script1`.
The main inputs of this function are:
- `filenums`: the number of parallel batches which should be launched on slurm
- `iters`: the number of div-enum iterations per batch  

Other inputs are used for personalizing the directories and filenames on the cluster.

After executing the script, the target directory should contain several bash files named `file_0.sh`, `file_1.sh` etc. depending on the `filenum` parameter that was provided.  
In addition, there should be one `runfiles.sh` file. This file contains the commands to submit the other files as job batches on the slurm cluster.

The results of a DEXOM run can be evaluated with the following scripts:  
- `dexom_cluster_results.py`compiles and removes duplicate solutions from the results of a parallel DEXOM run.  
- `pathway_enrichment.py` can be used to perform a pathway enrichment analysis using a one-sided hypergeometric test  
- `result_functions.py` contains the `plot_pca` function, which performs Principal Component Analysis on the enumeration solutions

## Examples

### Toy models
The `toy_models.py` script contains code for generating some small metabolic models and reaction weights.  
The toy_models folder contains some ready-to-use models and reaction weight files.  
The `main.py` script contains a simple example of the DEXOM workflow using one of the toy models.

### Recon 2.2
The example_data folder contains the model and the differential gene expression data which was used to test this new implementation.  
In order to produce reaction weights, you can call the `gpr_rules` script from the command line.  
This will create a file named "pval_0-01_reactionweights.csv" in the recon2v2 folder:  
```
python dexom_python/gpr_rules -m example_data/recon2v2_corrected.json -g example_data/pval_0-01_geneweights.csv -o example_data/pval_0-01_reactionweights
```
 
Then, call imat to produce a first context-specific subnetwork. This will create a file named "imat_solution.csv" in the example_data folder:  
```
python dexom_python/imat_functions.py -m example_data/recon2v2_corrected.json -r example_data/pval_0-01_reactionweights.csv -o example_data/imat_solution
```
To run DEXOM on a slurm cluster, call the enumeration.py script to create the necessary batch files (here: 100 batches with 100 iterations).   
Be careful to put the path to your installation of the CPLEX solver as the `-c` argument.   
This script assumes that you have cloned the `dexom-python` project on the cluster, which contains the `dexom_python` folder and the `example_data` folder in the same directory.  
Note that this step creates a file called "recon2v2_reactions_shuffled.csv", which shows the order in which rxn-enum will call the reactions from the model.  
```
python dexom_python/cluster_utils/write_cluster_scripts.py -m example_data/recon2v2_corrected.json -r example_data/pval_0-01_reactionweights.csv -p example_data/imat_solution.csv -o example_data/ -n 100 -i 100 -c /home/mstingl/save/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux
```
Then, submit the job to the slurm cluster.  
Note that if you created the files on a Windows pc, you must use the command `dos2unix runfiles.sh` before `sbatch runfiles.sh`:  
```
cd example_data/
sbatch runfiles.sh
cd ..
```
After all jobs are completed, you can analyze the results using the following scripts:  
```
python dexom_python/cluster_utils/dexom_cluster_results.py -i example_data/ -o example_data/ -n 100
python dexom_python/pathway_enrichment.py -s example_data/all_dexom_sols.csv -m example_data/recon2v2_corrected.json -o example_data/
python dexom_python/result_functions.py -s example_data/all_dexom_sols.csv -o example_data/
```
The file `all_dexom_sols.csv` contains all unique solutions enumerated with DEXOM.  
The file `output.txt` contains the average computation time per iteration and the proportion of duplicate solutions.  
The `.png` files contain boxplots of the pathway enrichment tests as well as a 2D PCA plot of the binary solution vectors.

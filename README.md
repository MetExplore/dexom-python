# DEXOM in python

This is a python implementation of DEXOM (Diversity-based enumeration of optimal context-specific metabolic networks)  
The original project, which was developped in MATLAB, can be found here: https://github.com/MetExplore/dexom  
Parts of the imat code were taken from the driven package for data-driven constraint-based analysis: https://github.com/opencobra/driven

## Requirements
- Python 3.7+
- CPLEX 12.10+

### Installing CPLEX

[Free license (Trial version)](https://www.ibm.com/analytics/cplex-optimizer): this version is limited to 1000 variables and 1000 constraints, and is therefore not useable on larger models

[Academic license](https://www.ibm.com/academic/technology/data-science): for this, you must sign up using an academic email address.
 - after logging in, you can access the download for "ILOG CPLEX Optimization Studio"
 - download version 12.10 or higher of the appropriate installer for your operating system
 - install the solver
 - update the PYTHONPATH environment variable by adding the directory containing the `setup.py` file appropriate for you OS and python version

### Python libraries
The python libraries needed to run the code can be found in the `requirements.txt` file.  
They can be installed using `pip install cobra`, `pip install numpy`, etc.

## Functions

These are the different functions which are available for context-specific network extraction

### iMAT
`imat.py` contains a modified version of the iMAT algorithm as defined by [(Shlomi et al. 2008)](https://pubmed.ncbi.nlm.nih.gov/18711341/).  
The main inputs of this algorithm are a model file, which must be supplied in a cobrapy-compatible format (SBML, JSON or MAT), and a reaction_weight file in which each reaction is attributed a score.  
These reaction weights must be determined prior to launching imat, using the GPR rules present in the metabolic model.
`model_functions.py` contains the `recon2_gpr` function, which transforms differential gene expression data into reaction weights for the recon 2 model, using the HGNC identifiers present in the model to connect the genes and reactions.

The remaining inputs of imat are:
- `epsilon`: the activation threshold of reactions with weight>0
- `threshold`: the activation threshold of all reactions
- `timelimit`: the solver time limit
- `feasibility`: the solver feasbility tolerance
- `mipgaptol`: the solver MIP gap tolerance
- `full`: a bool parameter for switching between the partial & full-DEXOM implementation

The partial implementation is the default version. In this version, binary flux indicator variables are created for each reaction with a non-zero weight.  
In the full-DEXOM implementation, binary flux indicator variables are created for every reaction in the model. This does not change the result of the imat function, but can be used for some of the enumeration methods below.

### enum_functions

Four methods for enumerating context-specific networks are available:
- `rxn-enum.py` contains reaction-enumeration
- `icut.py` contains integer-cut
- `maxdist.py` contains distance-maximization
- `div-enum.py` contains diversity-enumeration

An explanation of these methods can be found in [(Rodriguez-Mier et al. 2021)](https://doi.org/10.1371/journal.pcbi.1008730).  
Each of these methods can be used on its own. The same model and reaction_weights inputs must be provided as for the imat function.

New parameters for all 4 methods are:
- `prev_sol`: a starting imat solution
- `obj_tol`: a relative tolerance on the imat objective value for the optimality of the solutions  
icut, maxiter, and div-enum also have:
- `maxiter`: the maximum number of iterations to run
- `full`: set to True to use the full-DEXOM implementation  
As previously explained, the full-DEXOM implementation defines binary indicator variables for all reactions in the model. Although only the reactions with non-zero weights have an impact on the imat objective function, the distance maximization function which is used in maxdist and div-enum can make use of the binary indicators for all reactions. This increases the distance between the solutions, but requires significantly more computation time.  
maxdist and div-enum also have:
- `icut`: if True, an icut constraint will be applied to prevent duplicate solutions

## Parallelized DEXOM

The DEXOM algorithm is a combination of several network enumeration methods.  
`enumeration.py` contains the `write_batch_script1` function, which is used for creating a parallelization of DEXOM on a slurm computation cluster. 
The inputs of this function are:
- `filenums`: the number of parallel batches which should be launched on slurm
- `iters`: the number of div-enum iterations per batch

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
The recon2v2 folder containts the model and the differential gene expression data which was used to test this new implementation.  
In order to produce reaction weights, you can call the model_functions script from the command line. This will create a file named "pval_0-01_reactionweights.csv" in the recon2v2 folder:  
```
python src/model_functions -m recon2v2/recon2v2_corrected.json -g recon2v2/pval_0-01_geneweights.csv -o recon2v2/pval_0-01_reactionweights
```
 
Then, call imat to produce a first context-specific subnetwork. This will create a file named "imatsol.csv" in the recon2v2 folder:  
```
python src/imat -m recon2v2/recon2v2_corrected.json -r recon2v2/pval_0-01_reactionweights.csv -o recon2v2/imatsol
```
To run DEXOM on a slurm cluster, call the enumeration.py script to create the necessary batch files (here: 100 batches with 100 iterations). Be careful to use your own username after the `-u` input. This script assumes that you have cloned the `dexom_py` project into a `work` folder on the cluster, and that you have installed CPLEX v12.10 in the same `work` folder. Note that this step creates a file called "recon2v2_reactions_shuffled.csv", which shows the order in which rxn-enum will call the reactions from the model.  
```
python src/enum_functions/enumeration -m recon2v2/recon2v2_corrected.json -r recon2v2/pval_0-01_reactionweights.csv -p recon2v2/imatsol.csv -d recon2v2 -u mstingl -n 100 -i 100
```
Then, submit the job to the slurm cluster:  
```
cd recon2v2/
sbatch runfiles.sh
cd -
```
After all jobs are completed, you can analyze the results using the following scripts.  
```
python src/dexom_cluster_results -d recon2v2 -o recon2v2 -n 100
python src/pathway_enrichment -s recon2v2/all_dexom_sols.csv -m recon2v2/recon2v2_corrected.json -o recon2v2/pathways
```
This last step should produce the following files in the recon2v2 folder:
- all_outs.txt
- all_errs.txt
- all_divenum_res.csv
- all_rxn_enum_sols.csv
- all_div_enum_sols.csv
- all_dexom_sols.csv
- pathways_pvalues_over.csv
- pathways_pvalues_under.csv
- pathways_overrepresentation.png
- pathways_underrepresentation.png

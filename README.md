# DEXOM in python

This is a python implementation of DEXOM (Diversity-based enumeration of optimal context-specific metabolic networks)

The original project, which was developped in MATLAB, can be found here: https://github.com/MetExplore/dexom

Parts of the imat code were taken from the driven package for data-driven constraint-based analysis: https://github.com/opencobra/driven

## Requirements
- Python 3.7+
- CPLEX 12.10+

**Installing CPLEX**

[Free license (Trial version)](https://www.ibm.com/analytics/cplex-optimizer): this version is limited to 1000 variables and 1000 constraints, and is therefore not useable on larger models

[Academic license](https://www.ibm.com/academic/technology/data-science): for this, you must sign up using an academic email address.
 - after logging in, you can access the download for "ILOG CPLEX Optimization Studio"
 - download version 12.10 or higher
 - install the solver by executing the installer (.exe in Windows, .bin in Linux)
 - add the CPLEX directory to the PYTHONPATH environment variable


## Functions

These are the different functions which are available for context-specific network extraction

### iMAT
`imat.py` contains a modified version of the iMAT algorithm as defined by [(Shlomi et al. 2008)](https://pubmed.ncbi.nlm.nih.gov/18711341/).

In this implementation, instead of inputting raw gene expression data, the user inputs a reaction_weight file in which each reaction has already been attributed a score.

These reaction weights must be determined prior to launching imat, using the GPR rules present in the metabolic model.
`model_functions.py` contains the `recon2_gpr` function, which transforms differential gene expression data into reaction weights.

### enum_functions

Four methods for enumerating context-specific networks are available:
- `rxn-enum.py` contains reaction-enumeration
- `icut.py` contains integer-cut
- `maxdist.py` contains distance-maximization
- `div-enum.py` contains diversity-enumeration

## DEXOM

The DEXOM algorithm is a combination of several network enumeration methods.

`enumeration.py` contains the `write_batch_script1` function, which is used for creating a parallelization of DEXOM on a slurm computation cluster.

`dexom_cluster_results.py`compiles and removes duplicate solutions from the results of a parallel DEXOM run.

`pathway_enrichment.py` can be used to perform a pathway enrichment analysis using a one-sided hypergeometric test

`result_functions.py` contains the `plot_pca` function, which performs Principal Component Analysis on the enumeration solutions

### Toy examples
The `toy_models.py` script contains code for generating some small metabolic models and reaction weights.

The toy_models folder contains some ready-to-use models and reaction weight files.

The `main.py` script contains a simple example of the DEXOM workflow using one of the toy models.


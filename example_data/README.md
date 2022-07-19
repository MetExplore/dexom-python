# Example data

This folder contains a use case for the dexom algorithm.  
The `diff_gene_expression_data.csv` file contains DEGs which were obtained from RNA microarray data. It contains a t-score, logFC and a p-value.  
For our example presented in the README, we use the t-score to determine which genes have an increased or decreased expression.  
`pval_0-01_geneweights.csv` contains the same information as the previous file, but all genes with p-value > 0.01 were removed.  
`recon2v2_corrected.json` is a version of the recon2.2 model that is compatible with cobrapy.  

`recon2v2_reactions_subsystems.csv` contains, for each reaction in the model, the associated metabolic pathway.  
`recon2v2_subsystems_list.txt` contains a list of all metabolic pathways present in the model.  
These two files can be used when analyzing the DEXOM results.

import re
import argparse
import numpy as np
import pandas as pd
from symengine import sympify, Add, Mul, Max, Min
from dexom_python.model_functions import read_model, save_reaction_weights
pd.options.mode.chained_assignment = None


def replace_MulMax_AddMin(expression):
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (replace_MulMax_AddMin(arg) for arg in expression.args)
        if expression.__class__ == Mul:
            return Max(*replaced_args)
        elif expression.__class__ == Add:
            return Min(*replaced_args)
        else:
            return expression.func(*replaced_args)


def expression2qualitative(genes, column_list=None, proportion=0.25, method='keep', save=True,
                           outpath='geneweights'):
    """

    Parameters
    ----------
    genes: pandas.DataFrame
        dataframe with gene IDs in the index and gene expression values in a later column
    column_list: list
        column indexes containing gene expression values to be transformed. If empty, all columns will be transformed
    proportion: float
        proportion of genes to be used for determining high/low gene expression
    method: str
        one of "max", "mean" or "keep". chooses how to deal with genes containing multiple conflicting expression values
    save: bool
        if True, saves the resulting gene weights
    outpath: str
        if save=True, the .csv file will be saved to this path

    Returns
    -------
    gene_weights: a pandas DataFrame containing qualitative gene weights
    (-1 for low expression, 1 for high expression, 0 for in-between or no information)
    """
    genes = genes[genes.index == genes.index]  # eliminates NaN values
    if column_list is None:
        column_list = list(genes.columns)
    elif len(column_list) == 0:
        column_list = list(genes.columns)
    cutoff = 1/proportion
    colnames = genes[column_list]
    for col in colnames:
        if method == 'max':
            for x in set(genes.index):
                genes[col][x] = genes[col][x].max()
        elif method == 'mean':
            for x in set(genes.index):
                genes[col][x] = genes[col][x].mean()

        genes.sort_values(col, inplace=True)
        genes[col].iloc[:int(len(genes)//cutoff)] = -1.
        genes[col].iloc[int(len(genes)//cutoff):int(len(genes)*(cutoff-1)//cutoff)] = 0.
        genes[col].iloc[int(len(genes) * (cutoff-1) // cutoff):] = 1.
    for x in genes.index:
        if isinstance(x, float):
            genes.index = genes.index.astype(int)
            break
    if save:
        genes.to_csv(outpath+'.csv')
    return genes


def apply_gpr(model, gene_weights, save=True, filename='reaction_weights', duplicates='remove'):
    """
    Applies the GPR rules from a given metabolic model for creating reaction weights

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    gene_weights: dict or pd.Series
        a dictionary of pandas Series containing gene IDs & weights
    save: bool
        if True, saves the reaction weights as a csv file
    filename: str
        path where the file will be saved
    duplicates: str, any of "remove", "max", "min", "mean", "median"
        determines how to deal with genes presenting several expression values
    Returns
    -------
    reaction_weights: dict where keys = reaction IDs and values = weights
    """
    operations = {'min': np.min, 'max': np.max, 'mean': np.mean, 'median': np.median,
                  'remove': lambda x: x.mean() if len(x.value_counts()) == 1 else 0.}
    if isinstance(gene_weights, pd.Series):
        for gene in set(gene_weights.index):
            if isinstance(gene_weights[gene], pd.Series):
                vals = gene_weights.pop(gene)
                gene_weights[gene] = operations[duplicates](vals)
        gene_weights = gene_weights.to_dict()

    reaction_weights = {}
    gene_weight_dict = {}
    for k, v in gene_weights.items():
        if pd.isna(k):
            pass
        elif isinstance(k, float):
            # pandas library reads NCBI gene IDs as float, they must be converted to int, then str
            gene_weight_dict[str(int(k))] = float(v)
        else:
            gene_weight_dict[str(k)] = float(v)
    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            gen_list = [g.id for g in rxn.genes]
            expr_split = rxn.gene_reaction_rule.replace('(', '( ').replace(')', ' )').split()
            expr_split = ['g_' + re.sub(':|\.|-', '_', s) if s in gen_list else s for s in expr_split]
            new_weights = {'g_' + re.sub(':|\.|-', '_', g): gene_weight_dict.get(g, 0.) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0:
                    new_weights[g] = -v - 1e-15
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace(' or ', ' * ').replace(' and ', ' + ')
            weight = replace_MulMax_AddMin(sympify(expression)).subs(new_weights)
            if weight + 1e-15 in negweights:
                weight = -weight - 1e-15
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0.
    reaction_weights = {str(k): float(v) for k, v in reaction_weights.items()}
    if save:
        save_reaction_weights(reaction_weights, filename+'.csv')
    return reaction_weights


if __name__ == '__main__':

    description = 'Applies GPR rules to transform gene weights into reaction weights'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='GEM in json, sbml or matlab format')
    parser.add_argument('-g', '--gene_file', help='csv file containing gene identifiers and scores')
    parser.add_argument('-o', '--output', default='reaction_weights',
                        help='Path to which the reaction_weights .csv file is saved')
    parser.add_argument('--gene_ID', default='ID', help='column containing the gene identifiers')
    parser.add_argument('--gene_score', default='t', help='column containing the gene scores')
    parser.add_argument('-d', '--duplicates', default='remove', help='column containing the gene scores')
    parser.add_argument('--convert', action='store_true', help='converts gene expression to qualitative weights')
    parser.add_argument('-t', '--threshold', type=float, default=.25,
                        help='proportion of genes that are highly/lowly expressed (only used if --convert is selected)')
    args = parser.parse_args()

    model = read_model(args.model)

    genes = pd.read_csv(args.gene_file).set_index(args.gene_ID)
    if args.convert:
        genes = expression2qualitative(genes, column_list=[args.gene_score], proportion=args.threshold,
                                       outpath=args.output+'_qual_geneweights')
    gene_weights = pd.Series(genes[args.gene_score].values, index=genes.index)

    reaction_weights = apply_gpr(model=model, gene_weights=gene_weights, save=True, filename=args.output,
                                 duplicates=args.duplicates)

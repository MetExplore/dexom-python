import re
import argparse
import numpy as np
import pandas as pd
from warnings import warn
from symengine import sympify, Add, Mul, Max, Min, Pow, Symbol
from dexom_python.model_functions import read_model, save_reaction_weights
pd.options.mode.chained_assignment = None


def replace_MulMax_AddMin(expression):
    """
    Function used for parsing gpr expressions

    Parameters
    ----------
    expression: symengine expression
        a symengine/sympy expression of a gpr rule
    """
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (replace_MulMax_AddMin(arg) for arg in expression.args)
        if expression.__class__ == Mul:
            return Max(*replaced_args)
        elif expression.__class__ == Add:
            return Min(*replaced_args)
        elif expression.__class__ == Pow:
            return replace_MulMax_AddMin(expression.args[0])
        else:
            raise TypeError(f"Unsupported operation: {repr(expression)}")
            # return expression.func(*replaced_args)


def expression2qualitative(genes, column_list=None, proportion=0.25, method='keep', significant_genes='both',
                           save=True, outpath='geneweights'):
    """
    Transforms gene expression values/ gene scores into qualitative gene weights

    Parameters
    ----------
    genes: pandas.DataFrame or pandas.Series
        dataframe with gene IDs in the index and gene expression values in a later column
    column_list: list
        column indexes containing gene expression values to be transformed. If empty, all columns will be transformed
    proportion: tuple or float
        proportion of genes to be used for determining low and high gene expression
    method: str
        one of "max", "mean" or "keep". chooses how to deal with genes containing multiple conflicting expression values
    significant_genes: str
        one of "high", "low" or "both". chooses whether the conversion is applied only for the genes with
        highest expression, lowest epxression, or both
    save: bool
        if True, saves the resulting gene weights
    outpath: str
        if save=True, the .csv file will be saved to this path

    Returns
    -------
    gene_weights: a pandas DataFrame containing qualitative gene weights
        (-1 for low expression, 1 for high expression, 0 for in-between or no information)
    """
    if isinstance(genes, pd.Series):
        genes = pd.DataFrame(genes)
        column_list = list(genes.columns)
    genes = genes[genes.index == genes.index]  # eliminates NaN values in index
    if column_list is None:
        column_list = list(genes.columns)
    elif len(column_list) == 0:
        column_list = list(genes.columns)
    else:
        for col in column_list:
            if col not in genes.columns:
                raise KeyError('Column %s is not present in gene expression file' % col)
    if isinstance(proportion, float):
        lowthreshold = proportion
        highthreshold = 1-proportion
    else:
        lowthreshold, highthreshold = proportion
    for col in column_list:
        if method == 'max':
            for x in set(genes.index):
                genes[col][x] = genes[col][x].max()
        elif method == 'mean':
            for x in set(genes.index):
                genes[col][x] = genes[col][x].mean()
        genes.sort_values(col, inplace=True)
        genes[genes < genes.quantile(lowthreshold)] = -1.
        genes[(genes >= genes.quantile(lowthreshold)) & (genes < genes.quantile(highthreshold))] = 0.
        genes[genes >= genes.quantile(highthreshold)] = 1.
        if significant_genes == 'high':
            print('applying expression2qualitative only on genes with highest expression')
            genes[col][genes == -1.] = 0.
        elif significant_genes == 'low':
            print('applying expression2qualitative only on genes with lowest expression')
            genes[col][genes == 1.] = 0.
    for x in genes.index:
        if isinstance(x, float):
            genes.index = genes.index.astype(int)
            break
    if save:
        genes[column_list].to_csv(outpath+'.csv')
    return genes


def apply_gpr(model, gene_weights, save=True, filename='reaction_weights', duplicates='remove', null=0.):
    """
    Applies the GPR rules from a given metabolic model for creating reaction weights

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    gene_weights: dict or pandas.Series or pandas.DataFrame
        a dictionary/pandas Series containing gene IDs as keys/index & weights as values
    save: bool
        if True, saves the reaction weights as a csv file
    filename: str
        path where the file will be saved
    duplicates: str, any of "remove", "max", "min", "mean", "median"
        determines how to deal with genes presenting several expression values
    null: float
        value to return for reactions/genes with no information
    Returns
    -------
    reaction_weights: dict where keys = reaction IDs and values = weights
    """
    operations = {'min': np.min, 'max': np.max, 'mean': np.mean, 'median': np.median,
                  'remove': lambda x: x.mean() if len(x.value_counts()) == 1 else 0.}
    if isinstance(gene_weights, pd.DataFrame):
        print('The gene_weights argument was passed as a DataFrame, '
              'the reaction_weights will be saved separately and returned as a DataFrame')
        reaction_weights = []
        for condition in gene_weights:
            rw = apply_gpr(model=model, gene_weights=gene_weights[condition], save=True,
                           filename=filename+'_'+str(condition), duplicates=duplicates, null=null)
            reaction_weights.append(pd.Series(rw, name=str(condition)))
        rw = pd.concat(reaction_weights, axis=1)
        rw.to_csv(filename+'.csv', sep=';')
        return rw
    elif isinstance(gene_weights, pd.Series):
        gene_weights = gene_weights.loc[gene_weights.index.dropna()]
        for gene in set(gene_weights.index):
            if isinstance(gene_weights[gene], pd.Series):
                vals = gene_weights.pop(gene)
                gene_weights[gene] = operations[duplicates](vals)
        gene_weights = gene_weights.to_dict()
    if not isinstance(gene_weights, dict):
        raise TypeError('gene_weights must be a dictionary, pandas.Series or pandas.DataFrame')
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
            new_weights = {'g_' + re.sub(':|\.|-', '_', g): gene_weight_dict.get(g, null) for g in gen_list}
            expression = ' '.join(expr_split).replace(' or ', ' * ').replace(' and ', ' + ')
            weight = replace_MulMax_AddMin(sympify(expression)).subs(new_weights)
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = null
    reaction_weights = {str(k): float(v) for k, v in reaction_weights.items()}
    if save:
        save_reaction_weights(reaction_weights, filename+'.csv')
    return reaction_weights


def _main():
    """
    This function is called when you run this script from the commandline.
    It applies GPR rules to transform gene weights into reaction weights
    Use --help to see commandline parameters
    """
    description = 'Applies GPR rules to transform gene weights into reaction weights'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='GEM in json, sbml or matlab format')
    parser.add_argument('-g', '--gene_file', help='csv file containing gene identifiers and scores')
    parser.add_argument('-o', '--output', default='reaction_weights',
                        help='Path to which the reaction_weights .csv file is saved')
    parser.add_argument('--gene_ID', default='ID', help='column containing the gene identifiers')
    parser.add_argument('--gene_score', default=None, help='columns containing the gene scores, comma-separated')
    parser.add_argument('-d', '--duplicates', default='remove', help='column containing the gene scores')
    parser.add_argument('-n', '--null', type=float, default=0.,
                        help='value assigned to reactions/genes with no associated information')
    parser.add_argument('--convert', action='store_true', help='converts gene expression to qualitative weights')
    parser.add_argument('-t', '--threshold', default='0.25_0.75',
                        help='proportion of genes that are lowly/highly expressed (only used if --convert is selected)')
    parser.add_argument('-s', '--significant', default='both',
                        help='which genes have significant expression (either "high", "low" or "both", '
                             'only if --convert is selected)')
    args = parser.parse_args()

    model = read_model(args.model)
    genes = pd.read_csv(args.gene_file, sep=';|,|\t', engine='python').set_index(args.gene_ID)
    if args.gene_score is None:
        score_columns = list(genes.columns)
    else:
        score_columns = args.gene_score.split(',')

    if args.convert:
        proportion = args.threshold.split('_')
        if len(proportion) == 1:
            proportion = float(proportion[0])
        elif proportion[0] == '':
            proportion = (float(proportion[1]), 1.)
        elif proportion[1] == '':
            proportion = (0., float(proportion[1]))
        elif len(proportion) == 2:
            proportion = (float(proportion[0]), float(proportion[1]))
        else:
            ValueError('The threshold argument was provided in an incorrect format.')
        genes = expression2qualitative(genes=genes, column_list=score_columns, proportion=proportion, method='keep',
                                       significant_genes=args.significant, save=True,
                                       outpath=args.output+'_qual_geneweights')
    if len(score_columns) == 1:
        gene_weights = pd.Series(genes[score_columns[0]].values, index=genes.index)
    else:
        gene_weights = genes[score_columns].copy()
    reaction_weights = apply_gpr(model=model, gene_weights=gene_weights, save=True, filename=args.output,
                                 duplicates=args.duplicates, null=args.null)
    return True


if __name__ == '__main__':
    _main()

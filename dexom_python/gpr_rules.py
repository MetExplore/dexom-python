
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


def expression2qualitative(genes, column_list=[], proportion=0.25, method="keep", save=True,
                           outpath="geneweights"):
    """

    Parameters
    ----------
    expression: pandas.DataFrame
        dataframe with gene IDs in the index and gene expression values in a later column
    column_idx: list
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
    (-1 for low expression, 1 for high expression, 0 for in-between)
    """

    if not column_list:
        column_list = list(genes.columns)

    cutoff = 1/proportion
    colnames = genes[column_list]
    for col in colnames:
        if method == "max":
            for x in set(genes.index):
                genes[col][x] = genes[col][x].max()
        elif method == "mean":
            for x in set(genes.index):
                genes[col][x] = genes[col][x].mean()

        genes.sort_values(col, inplace=True)
        genes[col].iloc[:int(len(genes)//cutoff)] = -1.
        genes[col].iloc[int(len(genes)//cutoff):int(len(genes)*(cutoff-1)//cutoff)] = 0.
        genes[col].iloc[int(len(genes) * (cutoff-1) // cutoff):] = 1.
    if save:
        genes.to_csv(outpath+".csv")
    return genes


def prepare_expr_split_gen_list(rxn, modelname):
    """

    Parameters
    ----------
    rxn: cobra.Reaction
    modelname: str
        The name of the model. Currently only supports human1, recon1, recon2, iMM1865, zebrafish1

    Returns
    -------

    """
    if modelname == "recon2":
        expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
        expr_split = [s.replace(':', '_') if ':' in s else s for s in expr_split]
        rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
        gen_list = set([s.replace(':', '_') for s in rxngenes if ':' in s])
    elif modelname == "recon1":
        expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
        expr_split = ["g_" + s[:-4] if '_' in s else s for s in expr_split]
        rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
        gen_list = set(["g_" + s[:-4] for s in rxngenes if '_' in s])
    elif modelname == "iMM1865":
        expr_split = rxn.gene_reaction_rule.split()
        expr_split = ["g_" + s if s.isdigit() else s for s in expr_split]
        rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
        gen_list = set(["g_" + s for s in rxngenes if s.isdigit()])
    elif modelname == "human1":
        expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
        gen_list = set([g.id for g in rxn.genes])
    elif modelname == "zebrafish1":
        expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
        expr_split = [re.sub(':|\.|-', '_', s) for s in expr_split]
        gen_list = set([re.sub(':|\.|-', '_', g.id) for g in rxn.genes])
    elif modelname == "test":
        expr_split = [rxn.gene_reaction_rule]
        gen_list = set([g.id for g in rxn.genes])
    else:
        print("Modelname not found")
        expr_split = None
        gen_list = None

    return expr_split, gen_list


def apply_gpr(model, gene_weights, modelname, save=True, filename="reaction_weights"):
    """
    Applies the GPR rules from a given metabolic model for creating reaction weights

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    gene_weights: dict
        a dictionary containing gene IDs & weights
    modelname: str
        the name of the model
    save: bool
        if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction_weights: dict where keys = reaction IDs and values = weights
    """
    reaction_weights = {}
    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split, gen_list = prepare_expr_split_gen_list(rxn, modelname)
            new_weights = {g: float(gene_weights.get(g, 0)) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0:
                    new_weights[g] = -v - 1e-15
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace(' or ', ' * ').replace(' and ', ' + ')
            # weight = sympify(expression).xreplace({Mul: Max}).xreplace({Add: Min})
            weight = replace_MulMax_AddMin(sympify(expression)).subs(new_weights)
            if weight + 1e-15 in negweights:
                weight = -weight - 1e-15
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0.
    if save:
        save_reaction_weights(reaction_weights, filename+".csv")
    return reaction_weights


if __name__ == "__main__":

    description = "Applies GPR rules to transform gene weights into reaction weights"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="GEM in json, sbml or matlab format")
    parser.add_argument("-n", "--modelname", help="supported: human1, recon1, recon2, iMM1865, zebrafish1")
    parser.add_argument("-g", "--gene_file", help="csv file containing gene identifiers and scores")
    parser.add_argument("-o", "--output", default="reaction_weights",
                        help="Path to which the reaction_weights .csv file is saved")
    parser.add_argument("--gene_ID", default="ID", help="column containing the gene identifiers")
    parser.add_argument("--gene_score", default="t", help="column containing the gene scores")
    parser.add_argument("--convert", action='store_true', help="converts gene expression to qualitative weights")
    parser.add_argument("-t", "--threshold", type=float, default=.25,
                        help="proportion of genes that are highly/lowly expressed (only used if --convert is selected)")
    args = parser.parse_args()

    model = read_model(args.model)
    model_list = ["human1", "recon1", "recon2", "iMM1865", "zebrafish1"]

    genes = pd.read_csv(args.gene_file)
    genes.index = genes.pop(args.gene_ID)
    if args.convert:
        genes = expression2qualitative(genes, column_list=[args.gene_score], proportion=args.threshold,
                                       outpath=args.output+"_qual_geneweights")
    gene_weights = pd.Series(genes[args.gene_score].values, index=genes.index)

    # current behavior: all genes with several different weights are removed
    for x in set(gene_weights.index):
        if type(gene_weights[x]) != np.float64:
            if len(gene_weights[x].value_counts()) > 1:
                gene_weights.pop(x)
    gene_weights = gene_weights.to_dict()

    if args.modelname not in model_list:
        print("Unsupported model. The currently supported models are: human1, recon1, recon2, iMM1865, zebrafish1")
    else:
        reaction_weights = apply_gpr(model=model, gene_weights=gene_weights, modelname=args.modelname, save=True,
                                     filename=args.output)

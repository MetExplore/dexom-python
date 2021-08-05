
import six
import pandas as pd
import numpy as np
from symengine import Add, Mul, sympify, Max, Min
import re
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from pathlib import Path
import argparse


def clean_model(model, reaction_weights=None, full=False):
    """
    removes variables and constraints added to the model.solver during imat

    Parameters
    ----------
    model: cobra.Model
        a model that has previously been passed to imat
    reaction_weights: dict
        the same reaction weights used for the imat
    full: bool
        the same bool used for the imat calculation
    """
    if full:
        for rxn in model.reactions:
            rid = rxn.id
            if "x_"+rid in model.solver.variables:
                model.solver.remove(model.solver.variables["x_"+rid])
                model.solver.remove(model.solver.variables["xf_"+rid])
                model.solver.remove(model.solver.variables["xr_"+rid])
                model.solver.remove(model.solver.constraints["xr_"+rid+"_upper"])
                model.solver.remove(model.solver.constraints["xr_"+rid+"_lower"])
                model.solver.remove(model.solver.constraints["xf_"+rid+"_upper"])
                model.solver.remove(model.solver.constraints["xf_"+rid+"_lower"])
    else:
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0. and "rh_"+rid+"_pos" in model.solver.variables:
                model.solver.remove(model.solver.variables["rh_"+rid+"_pos"])
                model.solver.remove(model.solver.variables["rh_"+rid+"_neg"])
                model.solver.remove(model.solver.constraints["rh_"+rid+"_pos_bound"])
                model.solver.remove(model.solver.constraints["rh_"+rid+"_neg_bound"])
            elif weight < 0. and "rl_"+rid in model.solver.variables:
                model.solver.remove(model.solver.variables["rl_"+rid])
                model.solver.remove(model.solver.constraints["rl_"+rid+"_upper"])
                model.solver.remove(model.solver.constraints["rl_"+rid+"_lower"])


def load_reaction_weights(filename, rxn_names="reactions", weight_names="weights"):
    """
    loads reaction weights from a .csv file
    Parameters
    ----------
    filename: str
        the path + name of a .csv file containing reaction weights
    rxn_names: str
        the name of the column containing the reaction names
    weight_names: str
        the name of the column containing the weights

    Returns
    -------
    a dict of reaction weights
    """
    df = pd.read_csv(filename)
    df.index = df[rxn_names]
    reaction_weights = df[weight_names].to_dict()
    return {k: float(v) for k, v in reaction_weights.items() if float(v) == float(v)}


def save_reaction_weights(reaction_weights, filename):
    """
    Parameters
    ----------
    reaction_weights: dict
        a dictionary where keys = reaction IDs and values = weights
    filename: str
    Returns
    -------
    the reaction_weights dict as a pandas DataFrame
    """
    df = pd.DataFrame(reaction_weights.items(), columns=["reactions", "weights"])
    df.to_csv(filename)
    df.index = df["reactions"]
    return df["weights"]


def get_all_reactions_from_model(model, save=True, shuffle=False):
    """

    Parameters
    ----------
    model: a cobrapy model
    save: bool
        by default, exports the reactions in a csv format
    shuffle: bool
        set to True to shuffle the order of the reactions

    Returns
    -------
    A list of all reactions in the model
    """
    rxn_list = [r.id for r in model.reactions]
    if save:
        pd.Series(rxn_list).to_csv(model.id + "_reactions.csv", header=False, index=False)
    if shuffle:
        np.random.shuffle(rxn_list)
        pd.Series(rxn_list).to_csv(model.id + "_reactions_shuffled.csv", header=False, index=False)
    return rxn_list


def get_subsytems_from_model(model, save=True):
    """
    Creates a list of all subsystems of a model and their associated reactions
    Parameters
    ----------
    model: a cobrapy model
    save: bool

    Returns
    -------
    rxn_sub: a DataFrame with reaction names as index and subsystem name as column
    sub_list: a list of subsystems
    """
    rxn_sub = {}
    sub_list = []
    i = 0
    for rxn in model.reactions:
        rxn_sub[i] = (rxn.id, rxn.subsystem)
        i += 1
        if rxn.subsystem not in sub_list:
            sub_list.append(rxn.subsystem)
    if sub_list[-1] == "":
        sub_list.pop()
    rxn_sub = pd.DataFrame.from_dict(rxn_sub, orient="index", columns=["ID", "subsystem"])
    if save:
        rxn_sub.to_csv(model.id+"_reactions_subsystems.csv")
        with open(model.id+"_subsystems_list.txt", "w+") as file:
            file.write(";".join(sub_list))
    return rxn_sub, sub_list


def recon2_gpr(model, gene_file, genename="ID", genescore="t", save=True):
    """
    Applies the GPR rules from the recon2 model
    Parameters
    ----------
    model: a cobrapy model
    gene_file: the path to a csv file containing gene scores
    genename: the column containing the gene IDs
    genescore: the column containing the gene scores
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction weights: dict
    """
    reaction_weights = {}
    genes = pd.read_csv(gene_file)
    gene_weights = pd.DataFrame(genes[genescore])
    gene_weights.index = genes[genename]
    gene_weights = {idx.replace(':', '_'): np.max(gene_weights.loc[idx][genescore]) for idx in gene_weights.index}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
            expr_split = [s.replace(':', '_') if ':' in s else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set([s.replace(':', '_') for s in rxngenes if ':' in s])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0 and -v not in new_weights.values():
                    new_weights[g] = -v
                    negweights.append(-v)
                elif v < 0:
                    new_weights[g] = -v + 0.001
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace('or', '*').replace('and', '+')
            weight = sympify(expression, evaluate=False).replace(Mul, Max).replace(Add, Min).subs(new_weights, n=21)
            if weight - 0.001 in negweights:
                weight = -v + 0.001
            elif weight in negweights:
                weight = -weight
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0
    if save:
        save_reaction_weights(reaction_weights, "recon2_weights.csv")
    return reaction_weights


def recon1_gpr(model, gene_file, genename="ID", genescore="t", save=True):
    """
    Applies the GPR rules from the recon1 model
    Parameters
    ----------
    model: a cobrapy model
    gene_file: the path to a csv file containing gene scores
    genename: the column containing the gene IDs
    genescore: the column containing the gene scores
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction weights: dict
    """
    reaction_weights = {}

    genes = pd.read_csv(gene_file)
    gene_weights = pd.DataFrame(genes[genescore])
    gene_weights.index = genes[genename]
    gene_weights = {"g_"+str(idx): np.max(gene_weights.loc[idx][genescore]) for idx in gene_weights.index}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.replace("(", "( ").replace(")", " )").split()
            expr_split = ["g_"+s[:-4] if '_' in s else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set(["g_"+s[:-4] for s in rxngenes if '_' in s])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0 and -v not in new_weights.values():
                    new_weights[g] = -v
                    negweights.append(-v)
                elif v < 0:
                    new_weights[g] = -v + 0.001
                    negweights.append(-v)
            expression = ' '.join(expr_split).replace('or', '*').replace('and', '+')
            weight = sympify(expression, evaluate=False).replace(Mul, Max).replace(Add, Min).subs(new_weights, n=21)
            if weight - 0.001 in negweights:
                weight = -v + 0.001
            elif weight in negweights:
                weight = -weight
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0
    if save:
        save_reaction_weights(reaction_weights, "recon1_weights.csv")
    return reaction_weights


def iMM1865_gpr(model, gene_file, genename="ID", genescore="t", save=True):
    """
    Applies the GPR rules from the iMM1865 model
    Parameters
    ----------
    model: a cobrapy model
    gene_file: the path to a csv file containing gene scores
    genename: the column containing the gene IDs
    genescore: the column containing the gene scores
    save: if True, saves the reaction weights as a csv file

    Returns
    -------
    reaction weights: dict
    """
    reaction_weights = {}

    genes = pd.read_csv(gene_file)
    gene_weights = pd.DataFrame(genes[genescore])
    gene_weights.index = genes[genename]
    gene_weights = {"g_"+str(idx): np.max(gene_weights.loc[idx][genescore]) for idx in gene_weights.index}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.split()
            expr_split = ["g_"+s if s.isdigit() else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set(["g_"+s for s in rxngenes if s.isdigit()])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            negweights = []
            for g, v in new_weights.items():
                if v < 0 and -v not in new_weights.values():
                    new_weights[g] = -v
                    negweights.append(-v)
                elif v < 0:
                    new_weights[g] = -v + 0.001
                    negweights.append(-v)
            expression = " ".join(expr_split).replace("or", "*").replace("and", "+")
            weight = sympify(expression, evaluate=False).replace(Mul, Max).replace(Add, Min).subs(new_weights, n=21)
            if weight - 0.001 in negweights:
                weight = -v + 0.001
            elif weight in negweights:
                weight = -weight
            reaction_weights[rxn.id] = weight
        else:
            reaction_weights[rxn.id] = 0
    if save:
        save_reaction_weights(reaction_weights, "iMM1865_weights.csv")
    return reaction_weights


if __name__ == "__main__":
    ### the GPR result is rendered correct by modifying the _collapse_arguments function of the MinMaxBase class
    ### in /sympy/functions/elementary/miscellaneous.py
    ### see https://github.com/sympy/sympy/issues/21399 and https://github.com/sympy/sympy/pull/21547 for details

    description = "Applies GPR rules from the Recon 2.2 model and saves reaction weights as a csv file"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="recon 2.2 model in json, sbml or mat format")
    parser.add_argument("-g", "--gene_file", help="csv file containing gene HGNC identifiers and scores")
    parser.add_argument("--gene_ID", default="ID", help="column containing the gene HGNC identifiers")
    parser.add_argument("--gene_score", default="t", help="column containing the gene score to be used")
    args = parser.parse_args()

    fileformat = Path(args.model).suffix
    if fileformat == ".sbml" or fileformat == ".xml":
        model = read_sbml_model(args.model)
    elif fileformat == '.json':
        model = load_json_model(args.model)
    elif fileformat == ".mat":
        model = load_matlab_model(args.model)
    else:
        print("Only SBML, JSON, and Matlab formats are supported for the models")
        model = None

    genefile = pd.read_csv(args.gene_file)

    reaction_weights = recon2_gpr(model=model, gene_file=genefile, genename=args.gene_ID, genescore=args.gene_score,
                                  save=True)

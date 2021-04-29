
import six
import pandas as pd
from csv import DictReader, DictWriter
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from sympy import Add, Mul, sympify
from sympy.functions.elementary.miscellaneous import Max, Min
import re


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
        a dictionary where keys = reaction names and values = weights
    filename: str
    Returns
    -------
    the reaction_weights dict as a pandas DataFrame
    """
    df = pd.DataFrame(reaction_weights.items(), columns=["reactions", "weights"])
    df.to_csv(filename)
    df.index = df["reactions"]
    return df["weights"]


def human_weights_from_gpr(model, gene_file):
    reaction_weights = {}

    genes = pd.read_csv(gene_file, sep=",", decimal=",")
    gene_weights = pd.DataFrame(genes["t"])
    gene_weights.index = genes["entrez"]
    gene_weights = gene_weights["t"].to_dict()
    gene_weights = {"HGNC_"+str(k): float(v) for k, v in gene_weights.items()}

    for rxn in model.reactions:
        if "GENE ASSOCIATION" in rxn.notes:
            rxnnotes = rxn.notes["GENE ASSOCIATION"].replace(":", "_")
            expr_split = rxnnotes.split()
            rxnnotes = re.sub('and|or|\(|\)', '', rxnnotes)
            gen_list = set(rxnnotes.split())
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            expression = " ".join(expr_split).replace("or", "*").replace("and", "+")
            weight = sympify(expression, evaluate=False).replace(Mul, Max).replace(Add, Min).subs(new_weights, n=21)
            reaction_weights[rxn.id] = weight
    return reaction_weights


def mouse_weights_from_gpr(model, gene_file):
    reaction_weights = {}

    genes = pd.read_csv(gene_file, sep=",")
    gene_weights = pd.DataFrame(genes["Weight"])
    gene_weights.index = genes["ID"]
    gene_weights = gene_weights["Weight"].to_dict()
    gene_weights = {"g_"+str(k): float(v) for k, v in gene_weights.items()}

    for rxn in model.reactions:
        if len(rxn.genes) > 0:
            expr_split = rxn.gene_reaction_rule.split()
            expr_split = ["g_"+s if s.isdigit() else s for s in expr_split]
            rxngenes = re.sub('and|or|\(|\)', '', rxn.gene_reaction_rule).split()
            gen_list = set(["g_"+s for s in rxngenes if s.isdigit()])
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            expression = " ".join(expr_split).replace("or", "*").replace("and", "+")
            weight = sympify(expression, evaluate=False).replace(Mul, Max).replace(Add, Min).subs(new_weights, n=21)
            reaction_weights[rxn.id] = weight
    return reaction_weights


if __name__ == "__main__":

    # jul_model = read_sbml_model("recon2_2/Recon2.2_reimported2_test.xml")
    # # old_model = read_sbml_model("recon2_2/Recon2.2_Swainton2016.xml")
    # # mat_model = load_matlab_model("recon2_2/Recon2.2.mat")
    #
    # filename = "recon2_2/sign_MUvsWT_ids.csv"
    #
    # model = jul_model
    # rec_wei = human_weights_from_gpr(model, filename)
    #
    # pab_wei = load_reaction_weights("recon2_2/scores-pval-005.csv", "rxn", "rxnWeights")
    #
    # print("total reaction scores: ", len(rec_wei))
    # print("non-zero reaction scores: ", len(rec_wei)-list(rec_wei.values()).count(0))
    #
    # print("comparison with Pablo's file")
    # print("total reaction scores: ", len(pab_wei))
    # print("non-zero reaction scores: ", len(pab_wei)-list(pab_wei.values()).count(0))

    mouse = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    filename = "min_iMM_synthdata/imm1865_0.25_2.5_cholesterol.csv"
    rec_wei = mouse_weights_from_gpr(mouse, filename)

    pab_wei = load_reaction_weights("min_iMM_synthdata/imm1865_chol.csv", "Var1", "Var2")

    diff = {k: v-pab_wei[k] for k, v in rec_wei.items()}
    print(sum([abs(v) for v in diff.values()]))

    ### for the mouse model, the result is correct by modifying the _collapse_arguments function of the MinMaxBase class
    ### in /sympy/functions/elementary/miscellaneous.py
    # # factor out common elements as for



import six
import pandas as pd
from csv import DictReader, DictWriter
from cobra.io import load_json_model, read_sbml_model
from sympy import Add, Mul, sympify
from sympy.functions.elementary.miscellaneous import Max, Min


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


def load_reaction_weights(filename):
    """
    loads reaction weights from a .csv file
    Parameters
    ----------
    filename: str
        the path + name of a .csv file containing reaction weights with the following format:
        first row = reaction names, second row = weights

    Returns
    -------
    reaction_weights: dict
    """
    reaction_weights = {}

    with open(filename, newline="") as file:
        read = DictReader(file)
        for row in read:
            reaction_weights = row
    for k, v in reaction_weights.items():
        reaction_weights[k] = float(v)

    return reaction_weights


def create_weights_from_gpr(model, gene_file):
    reaction_weights = {}

    genes = pd.read_csv(gene_file, sep=";")
    gene_weights = pd.DataFrame(genes["t"])
    gene_weights.index = genes["entrez"]
    gene_weights = gene_weights["t"].to_dict()
    gene_weights = {"HGNC:"+str(k): float(v) for k, v in gene_weights.items()}

    for rxn in model.reactions:
        if len(rxn.genes) > 0 and any([g.id in gene_weights.keys() for g in rxn.genes]):
            expression = rxn.gene_expression()  # sympify(rxn.gene_reaction_rule)
            expression = expression.replace(Mul, Max).replace(Add, Min)
            gene_dict = {g.id: gene_weights.get(g.id) for g in rxn.genes}
            reaction_weights[rxn.id] = expression.subs(gene_dict).evalf()

    return gene_weights, reaction_weights


if __name__ == "__main__":

    jul_model = read_sbml_model("recon2_2/Recon2.2_reimported2.xml")
    # old_model = read_sbml_model("recon2_2/Recon2.2_Swainton2016.xml")
    filename = "recon2_2/recon2.2_tvals.csv"
    gen_wei, rec_wei = create_weights_from_gpr(jul_model, filename)
    print(gen_wei)
    print("------------------------")
    print(rec_wei)


import six
import pandas as pd
from csv import DictReader, DictWriter
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
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
    import re
    reaction_weights = {}

    genes = pd.read_csv(gene_file, sep=";")
    gene_weights = pd.DataFrame(genes["t"])
    gene_weights.index = genes["entrez"]
    gene_weights = gene_weights["t"].to_dict()
    gene_weights = {"HGNC_"+str(k): float(v) for k, v in gene_weights.items()}
    symbols = ["+", "*", "(", ")"]

    for rxn in model.reactions:
        if "GENE ASSOCIATION" in rxn.notes:
            rxnnotes = rxn.notes["GENE ASSOCIATION"].replace(":", "_")
            expr_split = rxnnotes.split()
            rxnnotes = re.sub('and|or|\(|\)', '', rxnnotes)
            gen_list = rxnnotes.split()
            new_weights = {g: gene_weights.get(g, 0) for g in gen_list}
            weight = sympify(" ".join(expr_split)).replace(Mul, Min).replace(Add, Max).subs(new_weights).evalf()
            reaction_weights[rxn.id] = weight

            # rxnnotes = rxn.notes["GENE ASSOCIATION"].replace('(', '( ').replace(')', ' )').replace("or", "+").replace(
            #     "and", "*").replace(":", "_")
            # expression_split = rxnnotes.split()
            # new_expression = rxnnotes.split()
            #if rxn.id == "FAOXC180x":

            # for temp in expression_split:
            #     if (temp not in symbols) and (temp not in gene_weights.keys()):
            #         idx = new_expression.index(temp)
            #         if len(new_expression) == 1:
            #             new_expression.pop()
            #         elif idx == len(new_expression)-1:
            #             new_expression.pop(-1)
            #             if len(new_expression) > 0:
            #                 new_expression.pop(-1)
            #         elif new_expression[idx + 1] == ")":
            #             new_expression.pop(idx)
            #             p = new_expression.pop(idx - 1)
            #             if p == "(":
            #                 new_expression.pop(idx - 1)
            #                 if new_expression:
            #                     new_expression.pop(idx - 2)
            #         else:
            #             new_expression.pop(idx)
            #             new_expression.pop(idx)
            # weight = 0.
            # if new_expression:
            #     weight = sympify(" ".join(new_expression)).replace(Mul, Min).replace(Add, Max).subs(gene_weights).evalf()
            # reaction_weights[rxn.id] = weight

    return gene_weights, reaction_weights


if __name__ == "__main__":
    jul_model = read_sbml_model("recon2_2/Recon2.2_reimported2_test.xml")

    # old_model = read_sbml_model("recon2_2/Recon2.2_Swainton2016.xml")
    # mat_model = load_matlab_model("recon2_2/Recon2.2.mat")

    filename = "recon2_2/sign_MUvsWT_ids_p01.csv"

    model = jul_model
    gen_wei, rec_wei = create_weights_from_gpr(model, filename)
    print(gen_wei)
    print("------------------------")
    print(rec_wei)

    comp_wei = load_reaction_weights("recon2_2/weights_pval-005.txt")

    print(len(rec_wei))
    print(list(rec_wei.values()).count(0))

    print("comparison")
    print(len(comp_wei))
    print(list(comp_wei.values()).count(0))

    # rxnnotes = "B:1 and (C:2 or D:4) or (E:5 and A:1)"
    # gene_weights = {"A:1": 1, "C:2": 2, "E:5": -3}
    # new_weights = {g.replace(":", "_"): gene_weights[g] for g in gene_weights.keys()}
    #
    # rxnnotes = rxnnotes.replace('(', '( ').replace(')', ' )').replace("or", "+").replace("and", "*").replace(":", "_")
    # expression_split = rxnnotes.split()
    # new_expression = rxnnotes.split()
    # symbols = ["+", "*", "(", ")"]
    # for temp in expression_split:
    #     if (temp not in symbols) and (temp not in new_weights.keys()):
    #         idx = new_expression.index(temp)
    #         if temp == expression_split[-1]:
    #             new_expression.pop(idx)
    #         elif new_expression[idx+1] == ")":
    #             new_expression.pop(idx)
    #             p = new_expression.pop(idx-1)
    #             if p == "(":
    #                 new_expression.pop(idx-1)
    #                 if new_expression:
    #                     new_expression.pop(idx-1)
    #         else:
    #             new_expression.pop(idx)
    #             new_expression.pop(idx)
    # weight = 0.
    # if new_expression:
    #     weight = sympify(" ".join(new_expression)).replace(Mul, Min).replace(Add, Max).subs(new_weights).evalf()
    # print(weight)

    ######
    #
    # expression_split = rxnnotes.split()
    # new_expression = rxnnotes.split()
    # for temp in expression_split:
    #     if temp != "or" and temp != "and":
    #         idx = new_expression.index(temp)
    #         if temp.strip("()") in gene_weights.keys():
    #             new_expression[idx] = temp.replace(temp.strip("()"), str(gene_weights[temp.strip("()")]))
    #         elif idx == 0:
    #             new_expression.pop(0)
    #             new_expression.pop(0)
    #             if "(" in temp:
    #                 new_expression[0] = '(' + new_expression[0]
    #         elif "(" in temp and ")" in temp:
    #             new_expression.pop(idx)
    #             new_expression.pop(idx)
    #         elif "(" in temp:
    #             new_expression.pop(idx)
    #             new_expression.pop(idx)
    #             new_expression[idx] = '(' + new_expression[idx]
    #         elif ")" in temp:
    #             new_expression.pop(idx)
    #             new_expression.pop(idx-1)
    #             new_expression[idx-2] = new_expression[idx-2] + ')'
    #         else:
    #             new_expression.pop(idx)
    #             new_expression.pop(idx)
    # expression = " ".join(new_expression)


import six
from csv import DictReader
import numpy as np
import pandas
from cobra import Solution
from csv import DictWriter


def clean_model(model, reaction_weights={}, full=False):
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


def write_solution(solution, threshold, filename):
    """
    writes an imat solution as a txt file
    Parameters
    ----------
    solution: cobra.Solution
    threshold: float
    filename: str
    """
    solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]

    with open(filename, "w+") as file:
        file.write(",fluxes,binary\n")
        for i, v in enumerate(solution.fluxes):
            file.write(solution.fluxes.index[i]+","+str(v)+","+str(solution_binary[i])+"\n")
        file.write("objective value: %f\n" % solution.objective_value)
        file.write("solver status: %s" % solution.status)


def read_solution(filename):

    df = pandas.read_csv(filename, index_col=0, skipfooter=2, engine="python")
    with open(filename, "r") as file:
        reader = file.read().split("\n")
        objective_value = float(reader[-2].split()[-1])
        status = reader[-1].split()[-1]
    solution = Solution(objective_value, status, df["fluxes"])
    binary = df["binary"].to_list()

    return solution, binary


if __name__=="__main__":
    from cobra.io import read_sbml_model
    #model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    df = pandas.read_csv("min_iMM1865/rxn_scores.csv", index_col=1)
    dict_p53 = df["3f"].to_dict()
    """
    with open('min_iMM1865/min_iMM1865_3f_weights.csv', 'w+', newline='') as csvfile:
        writer = DictWriter(csvfile, fieldnames=dict_p53.keys())
        writer.writeheader()
        writer.writerow(dict_p53)
    """


import six
from csv import DictReader, DictWriter
import numpy as np
import pandas as pd
from pathlib import Path
from cobra import Solution
import matplotlib.pyplot as plt


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
    Writes an imat solution as a txt file. The solution is written in a column format
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
    df = pd.read_csv(filename, index_col=0, skipfooter=2, engine="python")
    with open(filename, "r") as file:
        reader = file.read().split("\n")
        if reader[-1] == '':
            reader.pop(-1)
        objective_value = float(reader[-2].split()[-1])
        status = reader[-1].split()[-1]
    solution = Solution(objective_value, status, df["fluxes"])
    binary = df["binary"].to_list()

    return solution, binary


def write_dict_from_frame(df, out_file="dict.txt"):
    """
    When given a pandas DataFrame, writes it into a file as a dictionary (row 1: index, row 2: values, etc.)

    Parameters
    ----------
    df: pandas DataFrame or Series
    out_file: string
    """
    dictionary = df.to_dict()

    with open(out_file, 'w+', newline='') as csvfile:
        writer = DictWriter(csvfile, fieldnames=dictionary.keys())
        writer.writeheader()
        writer.writerow(dictionary)


def analyze_permutation(all_files, out_name="", sub_frame=None, sub_list=[]):
    """

    Parameters
    ----------
    all_files: Path or list of paths
        files containing imat binary solutions in rows
    out_name: string
        name of the output file (!! should not contain a suffix !!)
    sub_frame: pandas DataFrame or Series
        a series in which each reaction is associated to a pathway
    sub_list: list
        a list of all pathways present in the model

    Returns
    -------
    full_results: pandas DataFrame
        header: reaction names
        row 0: subsystem/pathway
        rows 1+: binary solutions
    """
    all_list = []
    if isinstance(sub_frame, pd.DataFrame):
        all_list.append(sub_frame)
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        all_list.append(df)

    full_results = pd.concat(all_list, axis=0, ignore_index=True)

    # sol_per_rxn = full_results[1:].sum()
    # sol_per_rxn /= len(full_results)-1
    #
    # # sub_list = full_results.T.agg(pd.unique)[0]
    # # sub_list = [x for x in subsystems if x==x]
    #
    # sol_per_path = {}
    # for sub in sub_list:
    #     rxns = full_results[full_results.isin([sub])].stack()[0].index
    #     sol_per_path[sub] = sol_per_rxn[rxns].sum()
    # sol_per_path = pd.Series(list(sol_per_path.values()), index=list(sol_per_path.keys()))

    # sol_per_rxn.to_csv("min_iMM1865/permutation_analysis/"+out_name+"_reactions.txt")
    # sol_per_path.to_csv("min_iMM1865/permutation_analysis/"+out_name+"_pathways.txt")
    # full_results.to_csv("p53_new_full.txt", index=0)
    return full_results


if __name__=="__main__":

    # df = pd.read_csv("min_iMM1865/rxn_scores.csv", index_col=1)
    # write_dict_to_frame(df["subsystem"], out_file = "min_iMM1865_subsystem.csv"

    subs = pd.read_csv("min_iMM1865/min_iMM1865_subsystem.csv")
    all_files = Path("min_iMM1865/perms_to_be_analyzed").glob("*.txt")

    with open("min_iMM1865/permutation_analysis/subsystems.txt", "r") as file:
        subsystems = file.read().split(";")

    full_results = analyze_permutation(all_files, out_name="p53_new", sub_frame=subs, sub_list=subsystems)

    hist_pathways = pd.DataFrame()

    solution, binary = read_solution("min_iMM1865/imat_p53.txt")
    newsolution, newbinary = read_solution("min_iMM1865/imat_p53_new.txt")

    binary = pd.Series(binary, index=solution.fluxes.index)
    newbinary = pd.Series(newbinary, index=newsolution.fluxes.index)

    sol_pathways = pd.Series(dtype=float)
    newsol_pathways = pd.Series(dtype=float)

    perms = len(full_results) - 1
    pvalues = pd.DataFrame(index=subsystems, columns=["normal", "-log10(p)"], dtype=float)
    newpvalues = pd.DataFrame(index=subsystems, columns=["normal", "-log10(p)"], dtype=float)

    for sub in subsystems:
        plt.clf()
        rxns = full_results[full_results.isin([sub])].stack()[0].index

        data = full_results[1:][rxns].sum(axis=1)
        hist_pathways[sub] = data.values
        sub = sub.replace("/", "")
        # data.hist(bins=np.arange(min(data), max(data) + 1)).get_figure().savefig("histograms/"+sub+".png")

        sol_pathways[sub] = binary[rxns].sum()
        newsol_pathways[sub] = newbinary[rxns].sum()

        pvalues["normal"][sub] = (sol_pathways[sub] - data.mean()) / data.std()
        newpvalues["normal"][sub] = (newsol_pathways[sub] - data.mean()) / data.std()

        temp = min(data[data <= sol_pathways[sub]].count() / perms, data[data >= sol_pathways[sub]].count() / perms)
        newtemp = min(data[data <= newsol_pathways[sub]].count() / perms, data[data >= newsol_pathways[sub]].count() / perms)
        if temp == 0.:
            temp = 0.001
        if newtemp == 0.:
            newtemp = 0.001
        pvalues["-log10(p)"][sub] = -np.log10(temp)
        newpvalues["-log10(p)"][sub] = -np.log10(newtemp)

    plt.clf()
    ax = pvalues.plot.scatter(0, 1, figsize=(20, 20))
    newax = newpvalues.plot.scatter(0, 1, figsize=(20, 20))
    for i in range(len(pvalues)):
        ax.annotate(pvalues.index[i], (pvalues["normal"][i], pvalues["-log10(p)"][i]))
        newax.annotate(newpvalues.index[i], (newpvalues["normal"][i], newpvalues["-log10(p)"][i]))
    ax.get_figure().savefig("scatterplot")
    newax.get_figure().savefig("scatterplot_new")

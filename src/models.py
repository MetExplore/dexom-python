
import six
from csv import DictReader, DictWriter
import numpy as np
import pandas as pd
from pathlib import Path
from cobra import Solution
import matplotlib.pyplot as plt


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


def analyze_permutation(perm_sols, imat_sol, out_path="permutation", sub_frame=None, sub_list=None):
    """

    Parameters
    ----------
    perm_sols: Path or list of paths
        files containing imat binary solutions in rows
    imat_sol: path or string
        path to an imat solution file created with write_solution()
    out_path: path or string
        path and name of output file
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
    for filename in perm_sols:
        df = pd.read_csv(filename, index_col=None, header=0)
        all_list.append(df)

    full_results = pd.concat(all_list, axis=0, ignore_index=True)

    solution, binary = read_solution(imat_sol)

    if not sub_list:
        sub_list = full_results.T.agg(pd.unique)[0]
        sub_list = [x for x in subsystems if x == x]  # removes nan values

    rxn_freq = full_results[1:].sum()
    rxn_freq /= len(full_results)-1

    path_max_act = {}
    for sub in sub_list:
        rxns = full_results[full_results.isin([sub])].stack()[0].index
        path_max_act[sub] = max(rxn_freq[rxns])
    path_max_act = pd.Series(list(path_max_act.values()), index=list(path_max_act.keys()))

    binary = pd.Series(binary, index=solution.fluxes.index)
    hist_pathways = pd.DataFrame()
    histograms = []
    sol_pathways = pd.Series(dtype=float)

    perms = len(full_results) - 1
    pvalues = pd.DataFrame(index=sub_list, columns=["normal", "-log10(p)"], dtype=float)

    for sub in sub_list:
        rxns = full_results[full_results.isin([sub])].stack()[0].index

        # create pathway histograms
        data = full_results[1:][rxns].sum(axis=1)
        hist_pathways[sub] = data.values
        subforsave = sub.replace("/", " ")
        plt.clf()
        fig = data.hist(bins=np.arange(min(data), max(data) + 1)).get_figure()
        histograms.append((fig, out_path+"_histogram "+subforsave+".png"))
        fig.savefig(out_path+"_histogram "+subforsave+".png")

        # count number of active reactions per pathway
        sol_pathways[sub] = binary[rxns].sum()

        # compute normalized active reactions per pathway & pvalues
        pvalues["normal"][sub] = (sol_pathways[sub] - data.mean()) / data.std()
        temp = min(data[data <= sol_pathways[sub]].count() / perms, data[data >= sol_pathways[sub]].count() / perms)
        if temp == 0.:
            temp = 0.001
        pvalues["-log10(p)"][sub] = -np.log10(temp)

    # create pvalue vulcanoplot
    plt.clf()
    pval_ax = pvalues.plot.scatter(0, 1, figsize=(20, 20))
    for i in range(len(pvalues)):
        pval_ax.annotate(pvalues.index[i], (pvalues["normal"][i], pvalues["-log10(p)"][i]))
    pval_fig = pval_ax.get_figure()
    pval_fig.savefig(out_path + "_scatterplot.png")

    # save files to path
    full_results.to_csv(out_path+"_all_solutions.txt", index=False)
    rxn_freq.to_csv(out_path+"_reaction_frequency.csv", sep=";")
    path_max_act.to_csv(out_path+"_pathway_maximal_frequency.csv", sep=";")

    sol_pathways.to_csv(out_path+"_pathways.csv", sep=";")
    pvalues.to_csv(out_path+"_pvalues.csv", sep=";")



    # for fig, filename in histograms:


    return full_results


if __name__ == "__main__":

    # df = pd.read_csv("min_iMM1865/rxn_scores.csv", index_col=1)
    # write_dict_to_frame(df["subsystem"], out_file = "min_iMM1865_subsystem.csv"

    all_files = Path("min_iMM1865/perms_to_be_analyzed").glob("*.txt")

    imat_sol = "min_iMM1865/imat_p53_new.txt"

    mypath = "permutation_new/p53_"

    subs = pd.read_csv("min_iMM1865/min_iMM1865_subsystem.csv")

    with open("min_iMM1865/subsystems.txt", "r") as file:
        subsystems = file.read().split(";")

    full_results = analyze_permutation(all_files, imat_sol, out_path=mypath, sub_frame=subs, sub_list=subsystems)

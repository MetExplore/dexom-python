
from csv import DictReader, DictWriter
import numpy as np
import pandas as pd
from pathlib import Path
from cobra import Solution
import matplotlib.pyplot as plt


def get_binary_sol(solution, threshold):
    binary = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]
    return binary


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
    When given a pandas DataFrame, writes it into a file as a dictionary (row 1: index, row 2: values)

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


def analyze_permutation(perm_sols, imat_sol, sub_frame=None, sub_list=None, savefiles=True, out_path="permutation"):
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
        fig = data.hist(bins=np.arange(min(data), max(data) + 2)).get_figure()
        histograms.append((fig, out_path+"_histogram "+subforsave+".png"))
        if savefiles:
            fig.savefig(out_path+"_histogram "+subforsave+".png")

        # count number of active reactions per pathway
        sol_pathways[sub] = binary[rxns].sum()

        # compute normalized active reactions per pathway & pvalues
        temp = data.std()
        if temp == 0.:
            temp = 1.
        pvalues["normal"][sub] = (sol_pathways[sub] - data.mean()) / temp

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
    if savefiles:
        pval_fig.savefig(out_path + "_scatterplot.png")

        # save files to path
        full_results.to_csv(out_path+"_all_solutions.txt", index=False)
        rxn_freq.to_csv(out_path+"_reaction_frequency.csv", sep=";")
        path_max_act.to_csv(out_path+"_pathway_maximal_frequency.csv", sep=";")

        sol_pathways.to_csv(out_path+"_pathways.csv", sep=";")
        pvalues.to_csv(out_path+"_pvalues.csv", sep=";")

    return full_results


if __name__ == "__main__":

    # df = pd.read_csv("min_iMM1865/rxn_scores.csv", index_col=1)
    # write_dict_to_frame(df["subsystem"], out_file = "min_iMM1865_subsystem.csv"

    all_files = Path("min_iMM1865/perms_to_be_analyzed").glob("*.txt")

    imat_sol = "min_iMM1865/imat_mp.txt"
    solution, binary = read_solution(imat_sol)

    mypath = "permutation/mp"

    subs = pd.read_csv("min_iMM1865/min_iMM1865_subsystem.csv")

    with open("min_iMM1865/subsystems.txt", "r") as file:
        subsystems = file.read().split(";")

    full_results = analyze_permutation(all_files, imat_sol, sub_frame=subs, sub_list=subsystems,
                                       savefiles=True, out_path=mypath)

    # all_list = []
    # for filename in all_files:
    #     df = pd.read_csv(filename, index_col=None, header=0)
    #     all_list.append(df)
    # weights = pd.concat(all_list, axis=0, ignore_index=True)
    # print(weights)
    # print(weights.drop_duplicates())

    # large_pathways = []
    # for sub in subsystems:
    #     temp = subs[subs.isin([sub])].stack().count()
    #     if temp >=15:
    #         large_pathways.append(sub)
    # print(large_pathways)

from csv import DictReader, DictWriter
import numpy as np
import pandas as pd
from pathlib import Path
from cobra import Solution
import matplotlib.pyplot as plt


def get_binary_sol(solution, threshold):
    binary = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]
    return binary


def write_solution(solution, threshold, filename="imat_sol.csv"):
    """
    Writes an optimize solution as a txt file. The solution is written in a column format
    Parameters
    ----------
    solution: cobra.Solution
    threshold: float
    filename: str
    """
    solution_binary = get_binary_sol(solution, threshold)

    with open(filename, "w+") as file:
        file.write("reaction,fluxes,binary\n")
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
    if type(perm_sols) == str:
        perm_sols = [perm_sols]
    for filename in perm_sols:
        df = pd.read_csv(filename, index_col=None, header=0)
        all_list.append(df)

    full_results = pd.concat(all_list, axis=0, ignore_index=True)

    solution, binary = read_solution(imat_sol)

    if not sub_list:
        sub_list = full_results.T.agg(pd.unique)[0]
        sub_list = [x for x in sub_list if x == x]  # removes nan values

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


def dexom_results(result_path, solution_path, out_path):

    res = pd.read_csv(result_path, index_col=0)
    df = pd.read_csv(solution_path, index_col=0)

    unique = len(df.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(df)-unique))

    time = res["time"].cumsum()
    print("Total computation time: %i s" % time.iloc[-1])
    print("Average time per iteration: %i s" % (time.iloc[-1]/len(df)))

    df = df.T
    avg_pairwise = []
    avg_near = []
    hammings = []
    h = 0
    for x in df:
        hammings.append([])
        if x > 0:
            for y in range(x):
                temp = sum(abs(df[x]-df[y]))
                h += temp
                hammings[x].append(temp)
                hammings[y].append(temp)
            avg_pairwise.append((h/(x*(x+1)/2))/len(df))
            temp = 0
            for v in hammings:
                temp += min(v)/len(df)
            avg_near.append(temp/x)
    x = range(len(avg_pairwise))

    plt.clf()
    plt.plot(x, avg_pairwise, 'r')
    plt.savefig(out_path + "_avg_pairwise.png")
    plt.clf()
    plt.plot(x, avg_near, 'g')
    plt.savefig(out_path + "_avg_nearest_neighbor.png")
    plt.clf()
    fig = time.plot().get_figure()
    fig.savefig(out_path + "_cumulated_time.png")
    plt.clf()
    fig = res["selected reactions"].plot().get_figure()
    fig.savefig(out_path + "_selected_reactions.png")
    return df.T


def pathway_histograms(solutions, sub_frame, sub_list, out_path):

    full_list = []
    hist_pathways = pd.DataFrame()
    histograms = []

    if isinstance(sub_frame, pd.DataFrame):
        full_list.append(sub_frame)
    if type(solutions) == str:
        solutions = [solutions]
    for filename in solutions:
        df = pd.read_csv(filename, index_col=0, header=0)
        mapp = {df.columns[i]: sub_frame.columns[i] for i in range(len(df.columns))}
        df.rename(mapp, axis=1, inplace=True)
        full_list.append(df)
    full_solutions = pd.concat(full_list, axis=0, ignore_index=True)

    for sub in sub_list:
        rxns = full_solutions[full_solutions.isin([sub])].stack()[0].index

        # create pathway histograms
        data = full_solutions[1:][rxns].sum(axis=1)
        hist_pathways[sub] = data.values
        subforsave = sub.replace("/", " ")
        plt.clf()
        fig = data.hist(bins=np.arange(min(data), max(data) + 2)).get_figure()
        histograms.append((fig, out_path+"_histogram "+subforsave+".png"))
        #fig.savefig(out_path+"_histogram "+subforsave+".png")

    return full_solutions


if __name__ == "__main__":
    ### permutation result analysis

    # all_files = Path("min_iMM1865/perms_to_be_analyzed").glob("*.txt")
    #
    # imat_sol = "min_iMM1865/imat_p53.txt"
    # solution, binary = read_solution(imat_sol)
    #
    # mypath = "permutation/"
    #
    # subs = pd.read_csv("min_iMM1865/min_iMM1865_subsystem.csv")
    #
    # with open("min_iMM1865/subsystems.txt", "r") as file:
    #     subsystems = file.read().split(";")
    #
    # full_results = analyze_permutation(all_files, imat_sol, sub_frame=subs, sub_list=subsystems,
    #                                    savefiles=True, out_path=mypath)
    #

    sols = "500 dexom/enum_dexom_solutions.csv"

    subs = pd.read_csv("min_iMM1865/min_iMM1865_subsystem.csv")

    with open("min_iMM1865/subsystems.txt", "r") as file:
        subsystems = file.read().split(";")

    sol = pathway_histograms(sols, subs, subsystems, "dexom_scatter/")
    print(sol)

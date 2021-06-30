
from csv import DictReader, DictWriter
import numpy as np
import pandas as pd
from pathlib import Path
from cobra import Solution
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact


def get_binary_sol(solution, threshold):
    binary = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]
    return binary


def get_obj_value_from_binary(binary, model, reaction_weights):
    wei = [reaction_weights.get(rxn.id, 0) for rxn in model.reactions]
    obj_val = sum([wei[i]*binary[i] if wei[i] >= 0 else -wei[i]*(1-binary[i]) for i in range(len(wei))])
    max_obj = sum([abs(x) for x in reaction_weights.values()])
    pos_wei = sum([1 for x in reaction_weights.values() if x > 0])
    pos_true = sum([1 for i in range(len(wei)) if wei[i] > 0 and binary[i] == 1])
    neg_wei = sum([1 for x in reaction_weights.values() if x < 0])
    neg_true = sum([1 for i in range(len(wei)) if wei[i] < 0 and binary[i] == 0])
    print("objective value: %.2f" % obj_val)
    print("maximal possible value: %.2f (%.2f%s, %.2f%s)" % (max_obj, 100*pos_true/pos_wei, "% RH",
                                                             100*neg_true/neg_wei, "% RL"))

    return obj_val


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


def read_solution(filename, model=None, reaction_weights=None):
    binary = True
    with open(filename, "r") as f:
        reader = f.read().split("\n")
        if reader[0] == "reaction,fluxes,binary":
            binary = False
            if reader[-1] == '':
                reader.pop(-1)
            objective_value = float(reader[-2].split()[-1])
            status = reader[-1].split()[-1]
    if binary:
        fluxes = pd.read_csv(filename, index_col=0).rename(index={0: "fluxes"}).T
        fluxes.index = [rxn.id for rxn in model.reactions]
        sol_bin = list(fluxes["fluxes"])
        objective_value = get_obj_value_from_binary(sol_bin, model, reaction_weights)
        status = "binary"
    else:
        df = pd.read_csv(filename, index_col=0, skipfooter=2, engine="python")
        fluxes = df["fluxes"]
        sol_bin = df["binary"].to_list()
    solution = Solution(objective_value, status, fluxes)
    return solution, sol_bin


def combine_solutions(sol_path):
    solutions = Path(sol_path).glob("*solutions.csv")
    sollist = []
    for sol in solutions:
        sollist.append(pd.read_csv(sol, index_col=0))
    fullsol = pd.concat(sollist, ignore_index=True)
    uniquesol = fullsol.drop_duplicates()
    print("There are %i unique solutions and %i duplicates." % (len(uniquesol), len(fullsol) - len(uniquesol)))
    uniquesol.to_csv(sol_path+"/combined_solutions.csv")
    return uniquesol



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
    elif type(sub_frame) == str:
        all_list.append(pd.read_csv(sub_frame, index_col=None))
    if type(perm_sols) == str:
        perm_sols = [perm_sols]
    for filename in perm_sols:
        df = pd.read_csv(filename, index_col=0, header=0)
        df.columns = sub_frame.columns
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
    # sol_pathways = pd.Series(dtype=float)
    sol_pathways = pd.DataFrame(index=sub_list, columns=["imat", "min", "max", "mean"], dtype=float)

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
        # sol_pathways[sub] = binary[rxns].sum()
        sol_pathways["imat"][sub] = binary[rxns].sum()
        sol_pathways["min"][sub] = min(data.values)
        sol_pathways["max"][sub] = max(data.values)
        sol_pathways["mean"][sub] = sum(data.values)/len(data.values)

        # compute normalized active reactions per pathway & pvalues
        temp = data.std()
        if temp == 0.:
            temp = 1.
        pvalues["normal"][sub] = (sol_pathways["imat"][sub] - data.mean()) / temp

        temp = min(data[data <= sol_pathways["imat"][sub]].count() / perms,
                   data[data >= sol_pathways["imat"][sub]].count() / perms)
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


def Fischer_pathways(solpath, subframe, sublist, outpath="Fischer_pathways.csv"):

    df = pd.read_csv(solpath, dtype=int, index_col=0).drop_duplicates(ignore_index=True)
    subframe = pd.read_csv(subframe, names=list(range(7785)))
    with open(sublist, "r") as file:
        sublist = file.read().split(";")

    rxn_list = []
    rxnnumber = {}
    for sub in sublist:
        rxns = list(subframe[subframe.isin([sub])].stack()[1].index)
        rxn_list.append(rxns)
        rxnnumber[sub] = len(rxns)

    sol_pathways = []
    for x in df.iterrows():
        sol_pathways.append([sum(x[1][r]) for r in rxn_list])

    pvals = {}
    for i, sub in enumerate(sublist):
        temp = []
        for sol in sol_pathways:
            table = np.array([[sol[i], len(rxn_list[i]) - sol[i]],
                              [sum(sol) - sol[i], 7784 - len(rxn_list[i]) - sum(sol) + sol[i]]])
            o, p = fisher_exact(table, alternative='greater')
            temp.append(-np.log10(p))
        pvals[sub] = temp
    newpvals = pd.DataFrame(pvals)
    newpvals.to_csv(outpath)
    return newpvals


if __name__ == "__main__":
    # result analysis
    Fischer_pathways("par_1_obj001_an/all_sol.csv", "recon2_2/recon2v2_reactions_subsystems.csv",
                     "recon2_2/recon2v2_subsystems_list.txt", "par_1_obj001_an/newobj_Fischer.csv")

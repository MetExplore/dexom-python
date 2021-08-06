
import argparse
from pathlib import Path
from cobra.io import load_json_model, load_matlab_model, read_sbml_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact
from matplotlib import rcParams
from statsmodels.stats.multitest import fdrcorrection
from model_functions import get_subsytems_from_model


def Fisher_pathways(solpath, subframe, sublist, outpath=""):
    """
    Performs pathway over- and underrepresentation analysis

    Parameters
    ----------
    solpath: file containing DEXOM solutions
    subframe: csv file associating reactions with subsystems
    sublist: list of subsystems
    outpath: path to which results are saved

    Returns
    -------
    csv files containing -log10 FDR-adjusted p-values
    """
    if out_path != "":
        out_path += "/"
    df = pd.read_csv(solpath, dtype=int, index_col=0)

    rxn_list = []
    rxnnumber = {}
    for sub in sublist:
        rxns = subframe.index[subframe["subsystem"] == sub].to_list()
        rxn_list.append(rxns)
        rxnnumber[sub] = len(rxns)

    sol_pathways = []
    for x in df.iterrows():
        sol_pathways.append([sum(x[1][r]) for r in rxn_list]+[sum(x[1])])

    pvalsu = {}
    pvalso = {}
    for i, sub in enumerate(sublist):
        tempu = []
        tempo = []
        for sol in sol_pathways:
            table = np.array([[sol[i], len(rxn_list[i]) - sol[i]],
                              [sol[-1] - sol[i], len(subframe) - len(rxn_list[i]) - sol[-1] + sol[i]]])
            o, pu = fisher_exact(table, alternative='less')
            o, po = fisher_exact(table, alternative='greater')
            tempu.append(pu)
            tempo.append(po)
        pvalsu[sub] = tempu
        pvalso[sub] = tempo
    over = pd.DataFrame(pvalso)
    under = pd.DataFrame(pvalsu)

    t, fdr = fdrcorrection(over.values.flatten())
    over = pd.DataFrame(-np.log10(fdr).reshape(over.shape), columns=over.columns)

    t, fdr = fdrcorrection(under.values.flatten())
    under = pd.DataFrame(-np.log10(fdr).reshape(under.shape), columns=under.columns)

    over.to_csv(outpath+"pathways_pvalues_over.csv")
    under.to_csv(outpath+"pathways_pvalues_under.csv")
    return over, under, fdr


def plot_Fisher_pathways(filename_over, filename_under, sublist, outpath="pathway_enrichment"):

    over = pd.read_csv(filename_over, index_col=0)
    under = pd.read_csv(filename_under, index_col=0)
    over.columns = sublist
    under.columns = sublist
    over = over.sort_index(axis=1, ascending=False)
    under = under.sort_index(axis=1, ascending=False)
    plt.clf()
    fig, ax = plt.subplots(figsize=(11, 20))
    rcParams['ytick.labelsize'] = 13
    over.boxplot(vert=False, widths=0.7)
    plt.plot(list(over.values)[0], ax.get_yticks(), 'ro')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.title("Overrepresentation analysis of active reactions per pathway", fontsize=15, loc='right', pad='20')
    plt.xlabel('-log10 adj. p-value', fontsize=15)
    plt.axvline(2.301, color="b")
    fig.savefig(outpath+"pathways_overrepresentation.png")

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 20))
    rcParams['ytick.labelsize'] = 13
    under.boxplot(vert=False, widths=0.7)
    plt.plot(list(under.values)[0], ax.get_yticks(), 'ro')
    plt.xticks(ticks=[10])
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.title("Underrepresentation analysis of active reactions per pathway", fontsize=15, loc='right', pad='20')
    plt.xlabel('-log10 adj. p-value', fontsize=15)
    plt.axvline(2.301, color="b")
    fig.savefig(outpath+"pathways_underrepresentation.png")

    return over, under


if __name__ == "__main__":
    description = "Performs pathway enrichment analysis using a hypergeometric test"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s", "--solutions", help="csv file containing enumeration solutions")
    parser.add_argument("-m", "--model", default=None, help="Metabolic model in sbml, json, or matlab format, "
                                                            "required if subframe & sublist are absent")
    parser.add_argument("--sublist", default=None, help="List of all pathways/subsystems in the model")
    parser.add_argument("--subframe", default=None, help="csv file assigning reactions to pathways/subsystemts")
    parser.add_argument("-o", "--out_path", default="", help="Path to which the output file is written")

    args = parser.parse_args()

    if args.model:
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
        subframe, sublist = get_subsytems_from_model(model, save=True, out_path=args.out_path)
    else:
        subframe = pd.read_csv(args.subframe, index_col=0)
        sublist = pd.read_csv(args.sublist, sep=";").columns.to_list()
    Fisher_pathways(solpath=args.solutions, subframe=subframe, sublist=sublist, outpath=args.out_path)
    plot_Fisher_pathways(filename_over=args.out_path+"pathways_pvalues_over.csv", sublist=sublist,
                         filename_under=args.out_path+"pathways_pvalues_under.csv", outpath=args.out_path)

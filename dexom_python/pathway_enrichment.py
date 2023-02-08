import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact
from matplotlib import rcParams
from statsmodels.stats.multitest import fdrcorrection
from dexom_python.model_functions import read_model, get_subsystems_from_model


def Fisher_groups(model, solpath, outpath='Fisher_groups'):
    """
    !!! This only works if the pathway name is stored in the model.groups property !!!
    For models where the pathways are stored in the model.reactions.subsystem property, use the Fischer_subsystems function

    Performs pathway over- and underrepresentation analysis

    Parameters
    ----------
    model: cobra.Model
    solpath: str
        file containing DEXOM solutions
    outpath: str
        path to which results are saved

    Returns
    -------
    over, under: pandas.DataFrames (saved as .csv files) containing -log10 BH-adjusted p-values
    """
    df = pd.read_csv(solpath, dtype=int, index_col=0)
    if df.columns[0] not in model.reactions:
        df.columns = [r.id for r in model.reactions]
    pvalsu = {}
    pvalso = {}
    for path in model.groups:
        tempu = []
        tempo = []
        for x in df.iterrows():
            sol = sum(x[1][r.id] for r in path.members)
            table = np.array([[sol, len(path.members) - sol],
                              [sum(x[1]) - sol, len(model.reactions) - len(path.members) - sum(x[1]) + sol]])
            o, pu = fisher_exact(table, alternative='less')
            o, po = fisher_exact(table, alternative='greater')
            tempu.append(pu)
            tempo.append(po)
        pvalsu[path.name] = tempu
        pvalso[path.name] = tempo
    over = pd.DataFrame(pvalso)
    under = pd.DataFrame(pvalsu)
    t, fdr = fdrcorrection(over.values.flatten())
    over = pd.DataFrame(-np.log10(fdr).reshape(over.shape), columns=over.columns)
    t, fdr = fdrcorrection(under.values.flatten())
    under = pd.DataFrame(-np.log10(fdr).reshape(under.shape), columns=under.columns)
    over.to_csv(outpath+'pathways_pvalues_over.csv')
    under.to_csv(outpath+'pathways_pvalues_under.csv')
    return over, under


def Fisher_subsystems(solpath, subframe, sublist, outpath='Fisher_subsystems'):
    """
    !!! This only works if the pathway name is stored in the model.reaction.subsystem property !!!
    For models where the pathways are stored in the model.groups property, use the Fischer_groups function

    Performs pathway over- and underrepresentation analysis

    Parameters
    ----------
    solpath: str
        file containing DEXOM solutions
    subframe: pandas.DataFrame
        dataframe with 'ID' column containing reaction IDs and one 'subsystem' column containing pathways
    sublist: list
        list of subsystems
    outpath: str
        path to which results are saved

    Returns
    -------
    over, under: pandas.DataFrames (saved as .csv files) containing -log10 BH-adjusted p-values
    """
    df = pd.read_csv(solpath, dtype=int, index_col=0)

    rxn_list = []
    rxnnumber = {}
    for sub in sublist:
        rxns = subframe.index[subframe['subsystem'] == sub].to_list()
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
    over.to_csv(outpath+'pathways_pvalues_over.csv')
    under.to_csv(outpath+'pathways_pvalues_under.csv')
    return over, under


def plot_Fisher_pathways(filename_over, filename_under, sublist, outpath='pathway_enrichment'):
    plt.ioff()
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
    plt.title('Overrepresentation analysis of active reactions per pathway', fontsize=15, loc='right', pad='20')
    plt.xlabel('-log10 adj. p-value', fontsize=15)
    plt.axvline(2.301, color='b')
    fig.savefig(outpath+'pathways_overrepresentation.png')

    plt.clf()
    fig, ax = plt.subplots(figsize=(11, 20))
    rcParams['ytick.labelsize'] = 13
    under.boxplot(vert=False, widths=0.7)
    plt.plot(list(under.values)[0], ax.get_yticks(), 'ro')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.title('Underrepresentation analysis of active reactions per pathway', fontsize=15, loc='right', pad='20')
    plt.xlabel('-log10 adj. p-value', fontsize=15)
    plt.axvline(2.301, color='b')
    fig.savefig(outpath+'pathways_underrepresentation.png')
    return over, under


def _main():
    """
    This function is called when you run this script from the commandline.
    It performs pathway enrichment analysis using a hypergeometric test (Fischer exact test)
    Use --help to see commandline parameters
    """
    description = 'Performs pathway enrichment analysis using a hypergeometric test (Fischer exact test)'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--solutions', help='csv file containing enumeration solutions')
    parser.add_argument('-m', '--model', default=None, help='Metabolic model in sbml, json, or matlab format, '
                                                            'required if subframe & sublist are absent')
    parser.add_argument('--sublist', default=None, help='List of all pathways/subsystems in the model')
    parser.add_argument('--subframe', default=None, help='csv file assigning reactions to pathways/subsystemts')
    parser.add_argument('-o', '--out_path', default='', help='Path to which the output file is written')
    args = parser.parse_args()

    groups = False
    if args.model is not None:
        model = read_model(args.model)
        if len(model.groups) > 0:
            groups = True
            sublist = [g.name for g in model.groups]
        else:
            subframe, sublist = get_subsystems_from_model(model, save=True, out_path=args.out_path)
    else:
        subframe = pd.read_csv(args.subframe, index_col=0)
        sublist = pd.read_csv(args.sublist, sep=';').columns.to_list()

    if groups:
        Fisher_groups(model=model, solpath=args.solutions, outpath=args.out_path)
    else:
        Fisher_subsystems(solpath=args.solutions, subframe=subframe, sublist=sublist, outpath=args.out_path)
    plot_Fisher_pathways(filename_over=args.out_path+'pathways_pvalues_over.csv', sublist=sublist,
                         filename_under=args.out_path+'pathways_pvalues_under.csv', outpath=args.out_path)
    return True


if __name__ == '__main__':
    _main()

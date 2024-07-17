import pandas as pd
import numpy as np
import argparse
import dexom_python as dp
from dexom_python.imat_functions import ImatException
from dexom_python.gpr_rules import apply_gpr
from dexom_python.default_parameter_values import DEFAULT_VALUES


def permute_genelabels(model, allgenes, geneindex=None, nperms=DEFAULT_VALUES['maxiter'], error_tol=DEFAULT_VALUES['maxiter']):
    """
    This function performs a permutation test in which the labels of gene expression values are randomly shuffled,
    and then an iMAT solution is computed with the new geneweights.

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    allgenes: pandas.Series
        index = gene IDs and values = expression values
    geneindex: pandas.Series or None
        index = model gene IDs and values = allgenes gene IDs
    nperms: int
        number of permutations to perform
    error_tol: int
        maximum number of consecutive failed iterations before interrupting the program

    Returns
    -------
    perm_solutions: list of imat solutions
    perm_binary: pandas.Dataframe of binary solutions
    perm_recs: pandas.Dataframe of reaction-weights
    perm_genes: pandas.Dataframe of permuted gene expression values
    """
    rng = np.random.default_rng()
    # geneweights = geneindex.replace(allgenes)
    if geneindex is not None:
        geneweights = geneindex.map(allgenes).dropna()
    else:
        geneweights = allgenes.dropna()

    reaction_weights = apply_gpr(model=model, gene_weights=geneweights, save=False)
    perm_genes = [geneweights.values]
    perm_recs = [reaction_weights.values()]
    perm_solutions = []
    perm_binary = []

    i = 0
    consecutive_errors = 0
    while i < nperms and consecutive_errors < error_tol:
        print('starting iteration %i' % (i + 1))
        rng.shuffle(allgenes.values)
        # gw = geneindex.replace(allgenes)
        if geneindex is not None:
            gw = geneindex.map(allgenes).dropna()
        else:
            gw = allgenes.dropna()
        reaction_weights = apply_gpr(model=model, gene_weights=gw, save=False)
        if reaction_weights in perm_recs:
            # if len(pd.concat([perm_recs, pd.Series(reaction_weights)], axis=1).T.drop_duplicates()) <= i:
            # if (perm_recs == pd.Series(reaction_weights)).all(axis=1).sum()>0:
            continue  # if the same reaction weights were already generated in a previous loop, skip this iteration
        try:
            solution = dp.imat(model, reaction_weights)
            thr = DEFAULT_VALUES['threshold']
            tol = DEFAULT_VALUES['tolerance']
            solution_binary = (np.abs(solution.fluxes) >= thr - tol).values.astype(int)
            perm_genes.append(gw.values)
            perm_recs.append(reaction_weights.values())
            perm_solutions.append(solution)
            perm_binary.append(solution_binary)
            i += 1
            consecutive_errors = 0
        except ImatException:
            print('imat in iteration %i failed' % (i+1))
            consecutive_errors += 1
    if consecutive_errors >= error_tol:
        print('Permutations aborted due to too many consecutive failed iterations. '
              'The results of the %i successful iterations will be returned now.' % len(perm_solutions))
    perm_genes = pd.DataFrame(perm_genes, columns=geneweights.index, index=['start']+list(range(len(perm_genes)-1))).T
    perm_recs = pd.DataFrame(perm_recs, columns=reaction_weights.keys(), index=['start']+list(range(len(perm_recs)-1))).T
    perm_binary = pd.DataFrame(perm_binary, columns=[r.id for r in model.reactions])
    return perm_solutions, perm_binary, perm_recs.drop('start', axis=1), perm_genes.drop('start', axis=1)


def _main():
    """
    This function is called when you run this script from the commandline.
    It performs the permutation test.
    Use --help to see commandline parameters
    """
    description = 'Performs the modified iMAT algorithm with reaction weights'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default=argparse.SUPPRESS,
                        help='Metabolic model in sbml, json, or matlab format')
    parser.add_argument('-g', '--gene_file', default=argparse.SUPPRESS,
                        help='csv file with gene IDs in the first column & gene expression values in the second')
    parser.add_argument('-n', '--npermutations', type=int, default=DEFAULT_VALUES['maxiter'],
                        help='Number of permutations to perform')
    parser.add_argument('-i', '--gene_index', default='false',
                        help='Define this parameter if your genefile uses different gene IDs than the model. '
                             'csv file in which the first column contains gene IDs from the metabolic model, '
                             'and the second column contains gene IDs from the genefile.')
    parser.add_argument('-o', '--output', default='perms', help='Path of the output file, without file format')
    parser.add_argument('-e', '--error_tol', type=int, default=DEFAULT_VALUES['maxiter'],
                        help='Maximum number of consecutive failed iterations before interrupting the program')
    args = parser.parse_args()
    model = dp.read_model(args.model)
    model = dp.check_model_options(model)

    gwfile = pd.read_csv(args.gene_file, sep=';|,|\t', engine='python')
    allgenes = gwfile[gwfile.columns[0]]

    if args.gene_index.lower() == 'false':
        geneindex = pd.Series(allgenes.index, index=allgenes.index)
    else:
        gifile = pd.read_csv(args.gene_index, sep=';|,|\t', engine='python')
        geneindex = gifile[gifile.columns[0]]

    n = args.npermutations
    e = args.error_tol

    sols, bins, recs, genes = permute_genelabels(model=model, allgenes=allgenes, geneindex=geneindex, nperms=n,
                                                 error_tol=e)
    bins.to_csv(args.output + '_solutions.csv')
    recs.to_csv(args.output + '_reactionweights.csv')
    genes.to_csv(args.output + '_geneweights.csv')
    flux = pd.concat([s.fluxes for s in sols], axis=1).T.reset_index(drop=True)
    flux.to_csv(args.output + '_fluxes.csv')
    return True


if __name__ == '__main__':
    _main()

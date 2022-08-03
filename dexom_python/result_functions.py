import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import warn
from cobra import Solution
from sklearn.decomposition import PCA


def write_solution(model, solution, threshold, filename='imat_sol.csv'):
    """
    Writes an optimize solution as a txt file. The solution is written in a column format
    Parameters
    ----------
    solution: cobra.Solution
    threshold: float
    filename: str
    """
    tol = model.solver.configuration.tolerances.feasibility
    solution_binary = (np.abs(solution.fluxes) >= threshold-tol).values.astype(int)
    with open(filename, 'w+') as file:
        file.write('reaction,fluxes,binary\n')
        for i, v in enumerate(solution.fluxes):
            file.write(solution.fluxes.index[i]+','+str(v)+','+str(solution_binary[i])+'\n')
        file.write('objective value: %f\n' % solution.objective_value)
        file.write('solver status: %s' % solution.status)
    return solution, solution_binary


def read_solution(filename, model=None):
    binary = True
    with open(filename, 'r') as f:
        reader = f.read().split('\n')
        if reader[0] == 'reaction,fluxes,binary':
            binary = False
            if reader[-1] == '':
                reader.pop(-1)
            objective_value = float(reader[-2].split()[-1])
            status = reader[-1].split()[-1]
    if binary:
        fluxes = pd.read_csv(filename, index_col=0).iloc[0]
        if model is not None:
            fluxes.index = [rxn.id for rxn in model.reactions]
        else:
            warn('A model is necessary for setting the reaction IDs in a binary solution.'
                 'Disregard this warning if the columns of the binary solution are already reaction IDs')
        sol_bin = np.array(fluxes.values)
        objective_value = 0.
        status = 'binary'
    else:
        df = pd.read_csv(filename, index_col=0, skipfooter=2, engine='python')
        fluxes = df['fluxes']
        sol_bin = df['binary'].to_list()
    solution = Solution(objective_value, status, fluxes)
    return solution, sol_bin


def combine_solutions(sol_path):
    solutions = Path(sol_path).glob('*solutions.csv')
    sollist = []
    for sol in solutions:
        sollist.append(pd.read_csv(sol, index_col=0))
    fullsol = pd.concat(sollist, ignore_index=True)
    uniquesol = fullsol.drop_duplicates()
    print('There are %i unique solutions and %i duplicates.' % (len(uniquesol), len(fullsol) - len(uniquesol)))
    uniquesol.to_csv(sol_path+'/combined_solutions.csv')
    return uniquesol


def plot_pca(solution_path, rxn_enum_solutions=None, save=True, save_name=''):
    """
    Plots a 2-dimensional PCA of enumeration solutions

    Parameters
    ----------
    solution_path: str
        csv file of enumeration solutions
    rxn_enum_solutions: str
        csv file of enumeration solutions. If specified, will plot these solutions in a different color
    save: bool
        if True, the pca-plot will be saved
    save_name: str
        name of the file to save

    Returns
    -------
    pca: sklearn.decomposition.PCA
    """
    X = pd.read_csv(solution_path, index_col=0)

    if rxn_enum_solutions is not None:
        X2 = pd.read_csv(rxn_enum_solutions, index_col=0)
        X_t = pd.concat([X, X2])
    else:
        X_t = X

    pca = PCA(n_components=2)
    pca.fit(X_t)

    comp = pca.transform(X)
    x = [c[0] for c in comp]
    y = [c[1] for c in comp]

    if rxn_enum_solutions is not None:
        comp2 = pca.transform(X2)
        x2 = [c[0] for c in comp2]
        y2 = [c[1] for c in comp2]

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=20)
    plt.ylabel('Principal Component 2', fontsize=20)
    plt.title('PCA of enumeration solutions', fontsize=20)
    if rxn_enum_solutions is not None:
        plt.scatter(x2, y2, color='g', label='rxn-enum solutions')
    plt.scatter(x, y, color='b', label='div-enum solutions')
    plt.scatter(x[0], y[0], color='r', label='iMAT solution')
    plt.legend(fontsize='large')
    if save:
        fig.savefig(save_name+'pca.png')
    return pca


if __name__ == '__main__':
    description = 'Plots a 2-dimensional PCA of enumeration solutions and saves as png'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--solutions', help='csv file containing diversity-enumeration solutions')
    parser.add_argument('-r', '--rxn_solutions', default=None,
                        help='(optional) csv file containing diversity-enumeration solutions')
    parser.add_argument('-o', '--out_path', default='', help='name of the file which will be saved')
    args = parser.parse_args()

    pca = plot_pca(args.solutions, rxn_enum_solutions=args.rxn_solutions, save_name=args.out_path)

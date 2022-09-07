import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dexom_python.result_functions import read_solution
from dexom_python.imat_functions import create_new_partial_variable_single, create_full_variable_single
from scipy.spatial.distance import pdist, squareform
from dexom_python.model_functions import DEFAULT_VALUES


class EnumSolution(object):
    """
    class for solutions of enumeration methods

    Parameters
    ----------
    solutions: list
        A list of pandas dataframes containing flux values with reaction ids as index
    binary: list
        A list containing binary arrays of reaction activity (0 for inactive, 1 for active)
    objective_value: float
        objective value returned by the solver at the end of the optimization
    """
    def __init__(self, solutions, binary, objective_value):
        self.solutions = solutions
        self.binary = binary
        self.objective_value = objective_value


def create_enum_variables(model, reaction_weights, eps=DEFAULT_VALUES['epsilon'], thr=DEFAULT_VALUES['threshold'],
                          full=False):
    for rid in reaction_weights.keys():
        if rid not in model.reactions:
            pass
        elif full and 'x_' + rid not in model.solver.variables:
            model = create_full_variable_single(model=model, rid=rid, reaction_weights=reaction_weights, epsilon=eps,
                                                threshold=thr)
        elif reaction_weights[rid] > 0 and 'rh_' + rid + '_pos' not in model.solver.variables:
            model = create_new_partial_variable_single(model=model, epsilon=eps, threshold=thr, rid=rid, pos=True)
        elif reaction_weights[rid] < 0 and 'rl_' + rid not in model.solver.variables:
            model = create_new_partial_variable_single(model=model, epsilon=eps, threshold=thr, rid=rid, pos=False)
    return model


def get_recent_solution_and_iteration(dirpath, startsol_num):
    """
    This functions fetches a solution from a given directory. The solutions are ordered by creation time, and one
    solution is picked using an exponential distribution (meaning that the most recent solution has the highest
    probability of being chosen)

    Parameters
    ----------
    dirpath: str
        a directory containing imat or enumeration solutions
    startsol_num: int
        the number of starting solutions present in the directory

    Returns
    -------
    solution: a Solution object
    iteration: int
        calculates the current iteration, based on how many solutions are already present in the folder
    """
    paths = sorted(list(Path(dirpath).glob('*solution*.csv')), key=os.path.getctime)
    paths.reverse()
    solpath = paths[int(np.random.exponential(5))]
    solution, binary = read_solution(solpath)
    iteration = len(paths) + 1 - startsol_num
    return solution, iteration


def combine_binary_solutions(sol_path, solution_pattern='*solutions*.csv', out_path=''):
    """
    Combines several binary solution files into one

    Parameters
    ----------
    sol_path: str
        folder in which binary solutions are saved
    solution_pattern: str
        pattern which is used to find binary solution files
    out_path: str
        path to which the combined solutions are saved

    Returns
    -------
    uniquesol: pandas.DataFrame
    """
    solutions = Path(sol_path).glob(solution_pattern)
    sollist = []
    for sol in solutions:
        sollist.append(pd.read_csv(sol, index_col=0))
    fullsol = pd.concat(sollist, ignore_index=True)
    uniquesol = fullsol.drop_duplicates()
    print('There are %i unique solutions and %i duplicates.' % (len(uniquesol), len(fullsol) - len(uniquesol)))
    uniquesol.to_csv(out_path+'combined_solutions.csv')
    return uniquesol


def analyze_div_enum_results(result_path, solution_path, out_path):
    """
    This function calculates the average pairwise hamming distance and average next neighbour distance
    for each iteration - it's very slow

    Parameters
    ----------
    result_path: csv results file from diversity-enum
    solution_path: csv solution file from diversity-enum
    out_path: path for saving

    Returns
    -------

    """
    res = pd.read_csv(result_path, index_col=0)
    sol = pd.read_csv(solution_path, index_col=0)
    unique = len(sol.drop_duplicates())
    print('There are %i unique solutions and %i duplicates' % (unique, len(sol)-unique))
    time = res['time'].cumsum()
    print('Total computation time: %i s' % time.iloc[-1])
    print('Average time per iteration: %i s' % (time.iloc[-1]/len(sol)))
    fig = time.plot().get_figure()
    fig.savefig(out_path + '_cumulated_time.png')
    plt.clf()
    fig = res['selected reactions'].plot().get_figure()
    fig.savefig(out_path + '_selected_reactions.png')
    sol = sol.drop_duplicates()
    avg_pairwise = []
    avg_nearest = []
    for i in range(2, len(sol) + 1):
        distances = pdist(sol[:i].values, metric='hamming')
        avg_pairwise.append(distances.mean())
        dist_mat = squareform(distances)
        avg_nearest.append(sum([min(x[np.nonzero(x)]) for x in dist_mat])/i)
    x = range(len(avg_pairwise))
    plt.clf()
    plt.plot(x, avg_pairwise, 'r')
    plt.savefig(out_path + '_avg_pairwise.png')
    plt.clf()
    plt.plot(x, avg_nearest, 'g')
    plt.savefig(out_path + '_avg_nearest_neighbor.png')
    plt.clf()
    fig = time.plot().get_figure()
    fig.savefig(out_path + '_cumulated_time.png')
    plt.clf()
    fig = res['selected reactions'].plot().get_figure()
    fig.savefig(out_path + '_selected_reactions.png')
    return sol.T

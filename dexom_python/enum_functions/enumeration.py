import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import warn
from scipy.spatial.distance import pdist, squareform
from dexom_python.result_functions import read_solution
from dexom_python.imat_functions import create_new_partial_variable_single, create_full_variable_single
from dexom_python.model_functions import DEFAULT_VALUES
from dexom_python.imat_functions import imat


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
        elif reaction_weights[rid] > 0:
            model = create_new_partial_variable_single(model=model, epsilon=eps, threshold=thr, rid=rid, pos=True)
        elif reaction_weights[rid] < 0:
            model = create_new_partial_variable_single(model=model, epsilon=eps, threshold=thr, rid=rid, pos=False)
    return model


def get_recent_solution_and_iteration(dirpath, startsol_num, solution_pattern='*solution_*.csv'):
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
    solution_pattern: str
        pattern which is used to find the solution files
    Returns
    -------
    solution: a Solution object
    iteration: int
        calculates the current iteration, based on how many solutions are already present in the folder
    """
    paths = sorted(list(Path(dirpath).glob(solution_pattern)), key=os.path.getctime)
    paths.reverse()
    idx = int(np.random.exponential(5))
    if idx > len(paths):
        idx = len(paths)-1
    solpath = paths[idx]
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
    fullsol.to_csv(out_path + 'combined_solutions.csv')
    uniquesol.to_csv(out_path + 'unique_solutions.csv')
    return uniquesol


def read_prev_sol(prev_sol_arg, model, rw, eps=DEFAULT_VALUES['epsilon'], thr=DEFAULT_VALUES['threshold'],
                  a=DEFAULT_VALUES['dist_anneal'], startsol=1, full=False, pattern='*solution_*.csv'):
    prev_sol_success = False
    if prev_sol_arg is not None:
        prev_sol_path = Path(prev_sol_arg)
        if prev_sol_path.is_file():
            prev_sol, prev_bin = read_solution(prev_sol_arg, model=model, solution_index=startsol)
            model = create_enum_variables(model, rw, eps=eps, thr=thr, full=full)
            prev_sol_success = True
        elif prev_sol_path.is_dir():
            try:
                prev_sol, i = get_recent_solution_and_iteration(prev_sol_arg, startsol_num=startsol,
                                                                solution_pattern=pattern)
                a = a ** i
                model = create_enum_variables(model, rw, eps=eps, thr=thr, full=full)
                prev_sol_success = True
            except IndexError:
                warn('Could not find any solutions in directory %s, computing new starting solution' % prev_sol_arg)
        else:
            warn('Could not read previous solution at path %s, computing new starting solution' % prev_sol_arg)
    if not prev_sol_success:
        prev_sol = imat(model, rw, epsilon=eps, threshold=thr)
    # if a binary solution was read, the optimal objective value must be calculated
    if prev_sol.objective_value < 0.:
        temp_sol = imat(model, rw, epsilon=eps, threshold=thr)
        prev_sol.objective_value = temp_sol.objective_value
    return prev_sol, a


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

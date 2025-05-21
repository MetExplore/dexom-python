import argparse
import time
import pandas as pd
import numpy as np
from symengine import sympify
from warnings import warn, catch_warnings, filterwarnings, resetwarnings
from cobra.exceptions import OptimizationError
from dexom_python.enum_functions.icut_functions import create_icut_constraint
from dexom_python.imat_functions import imat, create_optimality_constraint
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options, check_threshold_tolerance
from dexom_python.enum_functions.enumeration import EnumSolution, create_enum_variables, read_prev_sol, check_reaction_weights
from dexom_python.default_parameter_values import DEFAULT_VALUES


def create_maxdist_objective(model, reaction_weights, prev_sol, prev_sol_bin, only_ones=False, full=False):
    """
    Create the new objective for the maxdist algorithm.
    This objective is the minimization of similarity between the binary solution vectors
    If only_ones is set to False, the similarity will only be calculated with overlapping ones
    """
    expr = sympify('0')
    if full:
        for rxn in model.reactions:
            rid = rxn.id
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            x = model.solver.variables['x_' + rid]
            if prev_sol_bin[rid_loc] == 1:
                expr += x
            elif not only_ones:
                expr += 1 - x
    else:
        for rid, weight in reaction_weights.items():
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            if weight > 0:
                x = model.solver.variables['x_' + rid]
                if prev_sol_bin[rid_loc] == 1:
                    expr += x
                elif not only_ones:
                    expr -= x
            elif weight < 0:
                x_rl = model.solver.variables['x_' + rid] #sympify('1') -
                if prev_sol_bin[rid_loc] == 1:
                    expr -= - x_rl
                elif not only_ones:
                    expr += x_rl
    objective = model.solver.interface.Objective(expr, direction='min')
    return objective


def maxdist(model, reaction_weights, prev_sol=None, eps=DEFAULT_VALUES['epsilon'], thr=DEFAULT_VALUES['threshold'],
            obj_tol=DEFAULT_VALUES['obj_tol'], maxiter=DEFAULT_VALUES['maxiter'], icut=True, full=False, only_ones=False):
    """

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: imat Solution object
        an imat solution used as a starting point
    eps: float
        activation threshold in imat
    thr: float
        detection threshold of activated reactions
    obj_tol: float
        variance allowed in the objective_values of the solutions
    maxiter: foat
        maximum number of solutions to check for
    icut: bool
        if True, icut constraints are applied
    full: bool
        if True, carries out integer-cut on all reactions; if False, only on reactions with non-zero weights
    only_ones: bool
        if True, only the ones in the binary solution are used for distance calculation (as in dexom matlab)

    Returns
    -------
    solution: EnumSolution object
    """
    eps, thr = check_threshold_tolerance(model=model, epsilon=eps, threshold=thr)
    check_reaction_weights(reaction_weights)
    if prev_sol is None:
        prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=full)
    else:
        model = create_enum_variables(model=model, reaction_weights=reaction_weights, eps=eps, thr=thr, full=full)
    tol = model.solver.configuration.tolerances.feasibility
    icut_constraints = []
    all_solutions = [prev_sol]
    prev_sol_bin = (np.abs(prev_sol.fluxes) >= thr+tol).values.astype(int)
    all_binary = [prev_sol_bin]
    # adding the optimality constraint: the new objective value must be equal to the previous objective value
    opt_const = create_optimality_constraint(model, reaction_weights, prev_sol, obj_tol,
                                             name='maxdist_optimality', full=full)
    model.solver.add(opt_const)
    for idx in range(1, maxiter+1):
        t0 = time.perf_counter()
        if icut:
            # adding the icut constraint to prevent the algorithm from finding the same solutions
            const = create_icut_constraint(model, reaction_weights, epsilon=eps, threshold=thr, prev_sol=prev_sol, name='icut_'+str(idx), full=full)
            model.solver.add(const)
            icut_constraints.append(const)
        # defining the objective: minimize the number of overlapping ones and zeros
        objective = create_maxdist_objective(model, reaction_weights, prev_sol, prev_sol_bin, only_ones, full)
        model.objective = objective
        with catch_warnings():
            filterwarnings('error')
            try:
                # with model:
                prev_sol = model.optimize()
                all_solutions.append(prev_sol)
                all_binary.append(prev_sol_bin)
                t1 = time.perf_counter()
                print('time for iteration ' + str(idx) + ':', t1 - t0)
            except UserWarning as w:
                resetwarnings()
                prev_sol = all_solutions[-1]
                if 'time_limit' in str(w):
                    print('The solver has reached the timelimit in iteration %i. If this happens frequently, there may '
                          'be too many constraints in the model. Alternatively, you can try modifying solver '
                          'parameters such as the feasibility tolerance or the MIP gap tolerance.' % idx)
                    warn('Solver status is "time_limit" in iteration %i' % idx)
                elif 'infeasible' in str(w):
                    print('The solver has encountered an infeasible optimization in iteration %i. If this happens '
                          'frequently, there may be a problem with the starting solution. Alternatively, you can try '
                          'modifying solver parameters such as the feasibility tolerance or the MIP gap tolerance.'
                          % idx)
                    warn('Solver status is "infeasible" in iteration %i' % idx)
                else:
                    print('An unexpected error has occured during the solver call in iteration %i.' % idx)
                    warn(w)
            except OptimizationError as e:
                resetwarnings()
                prev_sol = all_solutions[-1]
                print('An unexpected error has occured during the solver call in iteration %i.' % idx)
                warn(str(e), UserWarning)
    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)
    return solution


def _main():
    """
    This function is called when you run this script from the commandline.
    It performs the distance-maximization enumeration algorithm
    Use --help to see commandline parameters
    """
    description = 'Performs the distance-maximization enumeration algorithm'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default=argparse.SUPPRESS,
                        help='Metabolic model in sbml, matlab, or json format')
    parser.add_argument('-r', '--reaction_weights', default=argparse.SUPPRESS,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-p', '--prev_sol', default=[], help='starting solution or directory of recent solutions')
    parser.add_argument('-e', '--epsilon', type=float, default=DEFAULT_VALUES['epsilon'],
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=DEFAULT_VALUES['threshold'],
                        help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=DEFAULT_VALUES['timelimit'], help='Solver time limit')
    parser.add_argument('--tol', type=float, default=DEFAULT_VALUES['tolerance'], help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=DEFAULT_VALUES['mipgap'], help='Solver MIP gap tolerance')
    parser.add_argument('--obj_tol', type=float, default=DEFAULT_VALUES['obj_tol'],
                        help='objective value tolerance, as a fraction of the original value')
    parser.add_argument('-i', '--maxiter', type=int, default=DEFAULT_VALUES['maxiter'], help='Iteration limit')
    parser.add_argument('-o', '--output', default='div_enum', help='Base name of output files, without format')
    parser.add_argument('--noicut', action='store_true', help='Use this flag to remove the icut constraint')
    parser.add_argument('--full', action='store_true', help='Use this flag to assign non-zero weights to all reactions')
    parser.add_argument('--onlyones', action='store_true', help='Use this flag for the old implementation of maxdist')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, tolerance=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights is not None:
        reaction_weights = load_reaction_weights(args.reaction_weights)
    prev_sol, _ = read_prev_sol(prev_sol_arg=args.prev_sol, model=model, rw=reaction_weights, eps=args.epsilon,
                                thr=args.threshold)
    icut = False if args.noicut else True
    maxdist_sol = maxdist(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol, eps=args.epsilon,
                          thr=args.threshold, obj_tol=args.obj_tol, maxiter=args.maxiter, icut=icut, full=args.full,
                          only_ones=args.onlyones)
    sol = pd.DataFrame(maxdist_sol.binary)
    sol.columns = [r.id for r in model.reactions]
    sol.to_csv(args.output+'_solutions.csv')
    fluxes = pd.concat([s.fluxes for s in maxdist_sol.solutions], axis=1).T.reset_index().drop('index', axis=1)
    fluxes.to_csv(args.output + '_fluxes.csv')
    return True


if __name__ == '__main__':
    _main()

import argparse
import six
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from warnings import warn
from symengine import sympify
from dexom_python.imat_functions import imat
from dexom_python.result_functions import read_solution, write_solution
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options, DEFAULT_VALUES
from dexom_python.enum_functions.enumeration import EnumSolution, get_recent_solution_and_iteration, create_enum_variables


def create_icut_constraint(model, reaction_weights, threshold, prev_sol, name, full=False):
    """
    Creates an icut constraint on the previously found solution.
    This solution is excluded from the solution space.
    """
    tol = model.solver.configuration.tolerances.feasibility
    if full:
        prev_sol_binary = (np.abs(prev_sol.fluxes) >= threshold-tol).values.astype(int)
        expr = sympify('1')
        newbound = sum(prev_sol_binary)
        cvector = [1 if x else -1 for x in prev_sol_binary]
        for idx, rxn in enumerate(model.reactions):
            expr += cvector[idx] * model.solver.variables['x_' + rxn.id]
    else:
        newbound = -1
        var_vals = []
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:
                y = model.solver.variables['rh_' + rid + '_pos']
                x = model.solver.variables['rh_' + rid + '_neg']
                if np.abs(prev_sol.fluxes[rid]) >= threshold-tol:
                    var_vals.append(y + x)
                    newbound += 1
                elif np.abs(prev_sol.fluxes[rid]) < threshold-tol:  # else
                    var_vals.append(-y - x)
            elif weight < 0:
                x = sympify('1') - model.solver.variables['rl_' + rid]
                if np.abs(prev_sol.fluxes[rid]) < (threshold-tol):
                    var_vals.append(x)
                    newbound += 1
                elif np.abs(prev_sol.fluxes[rid]) >= (threshold-tol):  # else
                    var_vals.append(-x)
        expr = sum(var_vals)
    constraint = model.solver.interface.Constraint(expr, ub=newbound, name=name)
    if expr.evalf() == 1:
        print('No reactions were found in reaction_weights when attempting to create an icut constraint')
        constraint = None
    return constraint


def icut(model, reaction_weights, prev_sol=None, eps=DEFAULT_VALUES['epsilon'], thr=DEFAULT_VALUES['threshold'],
         obj_tol=DEFAULT_VALUES['obj_tol'], maxiter=DEFAULT_VALUES['maxiter'], full=False):
    """
    integer-cut method

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
    full: bool
        if True, carries out integer-cut on all reactions; if False, only on reactions with non-zero weights
    Returns
    -------
    solution: EnumSolution object
        In the case of integer-cut, all_solutions and unique_solutions are identical
    """
    if prev_sol is None:
        prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=full)
    else:
        model = create_enum_variables(model=model, reaction_weights=reaction_weights, eps=eps, thr=thr, full=full)
    tol = model.solver.configuration.tolerances.feasibility
    prev_sol_binary = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
    optimal_objective_value = prev_sol.objective_value - obj_tol*prev_sol.objective_value

    all_solutions = [prev_sol]
    all_solutions_binary = [prev_sol_binary]
    icut_constraints = []

    for i in range(maxiter):
        t0 = time.perf_counter()
        const = create_icut_constraint(model, reaction_weights, thr, prev_sol, name='icut_'+str(i), full=full)
        model.solver.add(const)
        icut_constraints.append(const)
        try:
            prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=full)
        except:
            print('An error occured in iteration %i of icut, check if all feasible solutions have been found' % (i+1))
            break
        t1 = time.perf_counter()
        print('time for iteration '+str(i+1)+': ', t1-t0)
        if prev_sol.objective_value >= optimal_objective_value:
            all_solutions.append(prev_sol)
            prev_sol_binary = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
            all_solutions_binary.append(prev_sol_binary)
        else:
            break

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    solution = EnumSolution(all_solutions, all_solutions_binary, all_solutions[0].objective_value)
    if full:
        print('full icut iterations: ', i+1)
    else:
        print('partial icut iterations: ', i+1)
    return solution


def main():
    """
    This function is called when you run this script from the commandline.
    It performs the integer-cut enumeration algorithm
    Use --help to see commandline parameters
    """
    description = 'Performs the integer-cut enumeration algorithm'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, matlab, or json format')
    parser.add_argument('-r', '--reaction_weights', default=None,
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
    parser.add_argument('--full', action='store_true', help='Use this flag to assign non-zero weights to all reactions')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights is not None:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    prev_sol_success = False
    if args.prev_sol is not None:
        prev_sol_path = Path(args.prev_sol)
        if prev_sol_path.is_file():
            prev_sol, prev_bin = read_solution(args.prev_sol, model)
            model = create_enum_variables(model, reaction_weights, eps=args.epsilon, thr=args.threshold, full=args.full)
            prev_sol_success = True
        elif prev_sol_path.is_dir():
            try:
                prev_sol, i = get_recent_solution_and_iteration(args.prev_sol, args.startsol_num)
            except:
                warn('Could not find solution in directory %s, computing new starting solution' % args.prev_sol)
            else:
                model = create_enum_variables(model, reaction_weights, eps=args.epsilon, thr=args.threshold,
                                              full=args.full)
                prev_sol_success = True
        else:
            warn('Could not read previous solution at path %s, computing new starting solution' % args.prev_sol)
    if not prev_sol_success:
        prev_sol = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold)

    maxdist_sol = icut(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol, eps=args.epsilon,
                       thr=args.threshold, obj_tol=args.obj_tol, maxiter=args.maxiter, full=args.full)
    sol = pd.DataFrame(maxdist_sol.binary)
    sol.to_csv(args.output+'_solutions.csv')
    return True


if __name__ == '__main__':
    main()

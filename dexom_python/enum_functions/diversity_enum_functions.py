import argparse
import six
import time
import numpy as np
import pandas as pd
from warnings import catch_warnings, filterwarnings, resetwarnings, warn
from cobra.exceptions import OptimizationError
from dexom_python.imat_functions import imat
from dexom_python.result_functions import write_solution
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options, DEFAULT_VALUES
from dexom_python.enum_functions.enumeration import EnumSolution, create_enum_variables, read_prev_sol
from dexom_python.enum_functions.icut_functions import create_icut_constraint
from dexom_python.enum_functions.maxdist_functions import create_maxdist_constraint, create_maxdist_objective


def diversity_enum(model, reaction_weights, prev_sol=None, eps=DEFAULT_VALUES['epsilon'], thr=DEFAULT_VALUES['threshold'],
                   obj_tol=DEFAULT_VALUES['obj_tol'], maxiter=DEFAULT_VALUES['maxiter'],
                   dist_anneal=DEFAULT_VALUES['dist_anneal'], out_path='enum_dexom', icut=True, full=False, save=False):
    """
    diversity-based enumeration

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: Solution instance
        a previous imat solution
    eps: float
        activation threshold for highly expressed reactions
    thr: float
        detection threshold of activated reactions
    obj_tol: float
        variance allowed in the objective_values of the solutions
    maxiter: foat
        maximum number of solutions to search for
    dist_anneal: float
        parameter which influences the probability of selecting reactions
    out_path: str
        path to which the solutions are saved if save==True
    icut: bool
        if True, icut constraints are applied
    full: bool
        if True, the full-DEXOM implementation is used
    save: bool
        if True, every individual solution is saved in the iMAT solution format
    Returns
    -------
    solution: an EnumSolution object
    stats: a pandas.DataFrame containing the number of selected reactions and runtime of each iteration
    """

    primals = ['']
    constraints = ['']

    if prev_sol is None:
        prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=full)
    else:
        model = create_enum_variables(model=model, reaction_weights=reaction_weights, eps=eps, thr=thr, full=full)
    tol = model.solver.configuration.tolerances.feasibility
    times = []
    selected_recs = []
    prev_sol_bin = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
    all_solutions = [prev_sol]
    all_binary = [prev_sol_bin]
    icut_constraints = []
    # preserve the optimality of the original solution
    opt_const = create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol, 'dexom_optimality', full=full)
    model.solver.add(opt_const)
    for idx in range(1, maxiter+1):
        t0 = time.perf_counter()
        if icut:
            # adding the icut constraint to prevent the algorithm from finding duplicate solutions
            const = create_icut_constraint(model, reaction_weights, thr, prev_sol, 'icut_'+str(idx), full)
            model.solver.add(const)
            icut_constraints.append(const)
        # randomly selecting reactions with nonzero weights for the distance maximization step
        tempweights = {}
        i = 0
        for rid, weight in six.iteritems(reaction_weights):
            if np.random.random() > dist_anneal**idx and weight != 0:
                tempweights[rid] = weight
                i += 1
        selected_recs.append(i)
        objective = create_maxdist_objective(model, tempweights, prev_sol, prev_sol_bin, full=full)
        model.objective = objective
        t2 = time.perf_counter()
        print('time before optimizing in iteration ' + str(idx) + ':', t2 - t0)
        with catch_warnings():
            filterwarnings('error')
            try:
                with model:
                    prev_sol = model.optimize()
                    primals.append(pd.Series(model.solver.primal_values))
                    constraints.append(pd.Series(model.solver.constraint_values))
                prev_sol_bin = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
                all_solutions.append(prev_sol)
                all_binary.append(prev_sol_bin)
                if save:
                    write_solution(model, prev_sol, thr,
                                   filename=out_path+'_solution_'+time.strftime('%Y%m%d-%H%M%S')+'.csv')
                t1 = time.perf_counter()
                print('time for optimizing in iteration ' + str(idx) + ':', t1 - t2)
                times.append(t1 - t0)
            except UserWarning as w:
                resetwarnings()
                times.append(-1)
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
                times.append(-1)
                prev_sol = all_solutions[-1]
                print('An unexpected error has occured during the solver call in iteration %i.' % idx)
                warn(str(e), UserWarning)
    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)
    stats = pd.DataFrame({'selected reactions': selected_recs, 'time': times})
    sol = pd.DataFrame(solution.binary)
    if save:
        stats.to_csv(out_path+time.strftime('%Y%m%d-%H%M%S')+'_results.csv')
        sol.to_csv(out_path+time.strftime('%Y%m%d-%H%M%S')+'_solutions.csv')
    return solution, stats, primals, constraints


def main():
    """
    This function is called when you run this script from the commandline.
    It performs the reaction-enumeration algorithm on a specified list of reactions
    Use --help to see commandline parameters
    """
    description = 'Performs the diversity-enumeration algorithm'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, matlab, or json format')
    parser.add_argument('-r', '--reaction_weights', default=None,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-p', '--prev_sol', default=None, help='starting solution or directory of recent solutions')
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
    parser.add_argument('-a', '--dist_anneal', type=float, default=DEFAULT_VALUES['dist_anneal'],
                        help='this parameter 0<=a<=1 controls the distance between each successive solution, '
                             '0 meaning no distance and 1 maximal distance')
    parser.add_argument('-s', '--startsol', type=int, default=1, help='total number of starting solutions '
                                                                      '(if prev_sol is a directory)'
                                                                      'which solution to use as starting point'
                                                                      '(if prev_sol is a binary solution file)')
    parser.add_argument('--noicut', action='store_true', help='Use this flag to remove the icut constraint')
    parser.add_argument('--full', action='store_true', help='Use this flag to assign non-zero weights to all reactions')
    parser.add_argument('--save', action='store_true', help='Use this flag to save each individual solution')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights is not None:
        reaction_weights = load_reaction_weights(args.reaction_weights)
    icut = False if args.noicut else True
    prev_sol, dist_anneal = read_prev_sol(prev_sol_arg=args.prev_sol, model=model, rw=reaction_weights,
                                          eps=args.epsilon, thr=args.threshold, a=args.dist_anneal,
                                          startsol=args.startsol, full=args.full)

    dex_sol, dex_res = diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol,
                                      thr=args.threshold, maxiter=args.maxiter, obj_tol=args.obj_tol,
                                      dist_anneal=dist_anneal, out_path=args.output, icut=icut, full=args.full,
                                      save=args.save)
    dex_res.to_csv(args.output + '_results.csv')
    sol = pd.DataFrame(dex_sol.binary, columns=[r.id for r in model.reactions])
    sol.to_csv(args.output + '_solutions.csv')
    return True


if __name__ == '__main__':
    main()

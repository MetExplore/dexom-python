import argparse
import six
import time
import numpy as np
import pandas as pd
from dexom_python.imat_functions import imat
from dexom_python.result_functions import read_solution, write_solution
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options
from dexom_python.enum_functions.enumeration import EnumSolution, get_recent_solution_and_iteration, create_enum_variables
from dexom_python.enum_functions.icut_functions import create_icut_constraint
from dexom_python.enum_functions.maxdist_functions import create_maxdist_constraint, create_maxdist_objective


def diversity_enum(model, reaction_weights, prev_sol=None, eps=1e-3, thr=1e-5, obj_tol=1e-3, maxiter=10, dist_anneal=0.995,
                   out_path='enum_dexom', icut=True, full=False, save=False):
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
    """
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
        try:
            t2 = time.perf_counter()
            print('time before optimizing in iteration '+str(idx)+':', t2-t0)
            with model:
                prev_sol = model.optimize()
            prev_sol_bin = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
            all_solutions.append(prev_sol)
            all_binary.append(prev_sol_bin)
            if save:
                write_solution(model, prev_sol, thr,
                               filename=out_path+'_solution_'+time.strftime('%Y%m%d-%H%M%S')+'.csv')
            t1 = time.perf_counter()
            print('time for optimizing in iteration ' + str(idx) + ':', t1 - t2)
            times.append(t1 - t0)
        except:
            print('An error occured in iteration %i of dexom, no solution was returned' % idx)
            times.append(-1)
            prev_sol = all_solutions[-1]

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)
    df = pd.DataFrame({'selected reactions': selected_recs, 'time': times})
    sol = pd.DataFrame(solution.binary)
    if save:
        df.to_csv(out_path+time.strftime('%Y%m%d-%H%M%S')+'_results.csv')
        sol.to_csv(out_path+time.strftime('%Y%m%d-%H%M%S')+'_solutions.csv')
    return solution, df


if __name__ == '__main__':
    description = 'Performs the diversity-enumeration algorithm'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, matlab, or json format')
    parser.add_argument('-r', '--reaction_weights', default=None,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-p', '--prev_sol', default=[], help='starting solution or directory of recent solutions')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-3,
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=1e-5, help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=None, help='Solver time limit')
    parser.add_argument('-i', '--maxiter', type=int, default=10, help='Iteration limit')
    parser.add_argument('--tol', type=float, default=1e-8, help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=1e-6, help='Solver MIP gap tolerance')
    parser.add_argument('--obj_tol', type=float, default=1e-2,
                        help='objective value tolerance, as a fraction of the original value')
    parser.add_argument('-o', '--output', default='div_enum', help='Base name of output files, without format')
    parser.add_argument('-a', '--dist_anneal', type=float, default=0.995, help='annealing distance')
    parser.add_argument('-s', '--startsol_num', type=int, default=1, help='number of starting solutions'
                                                                          '(if prev_sol is a directory)')
    parser.add_argument('--noicut', action='store_true', help='Use this flag to remove the icut constraint')
    parser.add_argument('--full', action='store_true', help='Use this flag to assign non-zero weights to all reactions')
    parser.add_argument('--save', action='store_true', help='Use this flag to save each individual solution')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights is not None:
        reaction_weights = load_reaction_weights(args.reaction_weights)
    a = args.dist_anneal
    if '.' in args.prev_sol:
        prev_sol, prev_bin = read_solution(args.prev_sol, model, reaction_weights)
        model = create_enum_variables(model, reaction_weights, eps=args.epsilon, thr=args.threshold, full=args.full)
    elif args.prev_sol is not None:
        prev_sol, i = get_recent_solution_and_iteration(args.prev_sol, args.startsol_num)
        a = a ** i
        model = create_enum_variables(model, reaction_weights, eps=args.epsilon, thr=args.threshold, full=args.full)
    else:
        prev_sol = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold)
    icut = False if args.noicut else True
    save = True if args.save else False
    full = True if args.full else False

    dex_sol, dex_res = diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol,
                                      thr=args.threshold, maxiter=args.maxiter, obj_tol=args.obj_tol, dist_anneal=a,
                                      icut=icut, full=args.full, save=save)
    dex_res.to_csv(args.output + '_results.csv')
    dex_sol.to_csv(args.output + '_solutions.csv')

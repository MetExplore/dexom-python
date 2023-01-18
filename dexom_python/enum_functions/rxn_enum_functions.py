import argparse
import os
import pandas as pd
import numpy as np
from dexom_python.imat_functions import imat, ImatException
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options, DEFAULT_VALUES
from dexom_python.result_functions import write_solution
from dexom_python.enum_functions.enumeration import create_enum_variables, read_prev_sol
from warnings import warn, filterwarnings, catch_warnings, resetwarnings


class RxnEnumSolution(object):
    def __init__(self, all_solutions, unique_solutions, all_binary, unique_binary,
                 all_reactions=None, unique_reactions=None, objective_value=-1.):
        self.all_solutions = all_solutions
        self.unique_solutions = unique_solutions
        self.all_binary = all_binary
        self.unique_binary = unique_binary
        self.all_reactions = all_reactions
        self.unique_reactions = unique_reactions
        self.objective_value = objective_value


def rxn_enum(model, reaction_weights, prev_sol=None, rxn_list=None, eps=DEFAULT_VALUES['epsilon'],
             thr=DEFAULT_VALUES['threshold'], obj_tol=DEFAULT_VALUES['obj_tol'], out_path='enum_rxn', save=False):
    """
    Reaction enumeration method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: imat Solution object
        an imat solution used as a starting point
    rxn_list: list
        a list of reactions on which reaction-enumeration will be performed. By default, all reactions are used
    eps: float
        activation threshold in imat
    thr: float
        detection threshold of activated reactions
    obj_tol: float
        variance allowed in the objective_values of the solutions
    out_path: str
        path to which the solutions are saved if save==True
    save: bool
        if True, every individual solution is saved in the iMAT solution format
    Returns
    -------
    solution: RxnEnumSolution object
    """
    if prev_sol is None:
        prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=False)
    else:
        model = create_enum_variables(model=model, reaction_weights=reaction_weights, eps=eps, thr=thr, full=False)
    tol = model.solver.configuration.tolerances.feasibility
    prev_sol_bin = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
    optimal_objective_value = prev_sol.objective_value - prev_sol.objective_value * obj_tol

    all_solutions = [prev_sol]
    all_solutions_binary = [prev_sol_bin]
    unique_solutions = [prev_sol]
    unique_solutions_binary = [prev_sol_bin]
    all_reactions = []  # for each solution, save which reaction was activated/inactived by the algorithm
    unique_reactions = []
    if save:  # when saving each individual solution, ensure that the out_path is a directory
        os.makedirs(out_path, exist_ok=True)
        if out_path[-1] not in ('\\', '/'):
            out_path += os.sep
    if rxn_list is None:
        rxns = list(model.reactions)
        rxn_list = [r.id for r in rxns]
    for rid in rxn_list:
        if rid not in model.reactions:
            print('The following reaction ID was not found in the model: %s' % rid)
            continue
        idx = np.where(prev_sol.fluxes.index == rid)[0][0]
        with model as model_temp:
            if rid in model.reactions:
                rxn = model_temp.reactions.get_by_id(rid)
                # for active fluxes, check inactivation
                if prev_sol_bin[idx] == 1:
                    rxn.bounds = (0., 0.)
                # for inactive fluxes, check activation
                else:
                    upper_bound_temp = rxn.upper_bound
                    # for inactive reversible fluxes, check activation in backwards direction
                    if rxn.lower_bound < 0.:
                        # with catch_warnings():
                        #     filterwarnings('error')
                        try:
                            rxn.upper_bound = -thr
                            temp_sol = imat(model_temp, reaction_weights, epsilon=eps, threshold=thr)
                            temp_sol_bin = (np.abs(temp_sol.fluxes) >= thr-tol).values.astype(int)
                            if temp_sol.objective_value >= optimal_objective_value:
                                all_solutions.append(temp_sol)
                                all_solutions_binary.append(temp_sol_bin)
                                if not np.any(np.all(temp_sol_bin == unique_solutions_binary, axis=1)):
                                    unique_solutions.append(temp_sol)
                                    unique_solutions_binary.append(temp_sol_bin)
                                    unique_reactions.append(rid+'_backwards')
                                    if save:
                                        filename = out_path+'_solution_'+str(len(unique_solutions)-1)+'.csv'
                                        write_solution(model, temp_sol, thr, filename)
                        except ImatException as w:
                            if 'time_limit' in str(w):
                                print('The solver has reached the timelimit for reaction %s_reverse. If this '
                                      'happens frequently, there may be too many constraints in the model. '
                                      'Alternatively, you can try modifying solver parameters such as the '
                                      'feasibility tolerance or the MIP gap tolerance.' % rid)
                                warn('Solver status is "time_limit" with reaction %s_reverse' % rid)
                            elif 'feasibility' in str(w):
                                print('The solver has encountered an infeasible optimization with reaction '
                                      '%s_reverse. The model may be infeasible when this reaction is '
                                      'irreversible. If this happens frequently, there may be a problem with '
                                      'the starting solution, or the tolerance parameters.' % rid)
                                warn('Solver status is "infeasible" when reaction %s_reverse is irreversible' % rid)
                            else:
                                print('An unexpected error has occured during the solver call with reaction '
                                      '%s_reverse.' % rid)
                                warn(str(w))
                        finally:
                            rxn.upper_bound = upper_bound_temp
                    # for all inactive fluxes, check activation in forwards direction
                    if rxn.upper_bound >= thr:
                        rxn.lower_bound = thr
                    else:
                        print('reaction %s has an upper bound below the detection limit, it cannot carry flux.' % rid)
                        rxn.lower_bound = rxn.upper_bound
                        continue
                # for all fluxes: compute solution with new bounds
                # with catch_warnings():
                #     filterwarnings('error')
                try:
                    temp_sol = imat(model_temp, reaction_weights, epsilon=eps, threshold=thr)
                    if temp_sol is None:
                        print('this print should not be reached')
                    #     warn('this warning should not be reached')
                    temp_sol_bin = (np.abs(temp_sol.fluxes) >= thr-tol).values.astype(int)
                    if temp_sol.objective_value >= optimal_objective_value:
                        all_solutions.append(temp_sol)
                        all_solutions_binary.append(temp_sol_bin)
                        all_reactions.append(rid)
                        if not np.any(np.all(temp_sol_bin == unique_solutions_binary, axis=1)):
                            unique_solutions.append(temp_sol)
                            unique_solutions_binary.append(temp_sol_bin)
                            unique_reactions.append(rid)
                            if save:
                                filename = out_path+'_solution_'+str(len(unique_solutions)-1)+'.csv'
                                write_solution(model, temp_sol, thr, filename)
                except ImatException as w:
                    if 'time_limit' in str(w):
                        print('The solver has reached the timelimit for reaction %s. If this happens frequently, '
                              'there may be too many constraints in the model. Alternatively, you can try '
                              'modifying solver parameters such as the feasibility tolerance or the MIP gap '
                              'tolerance.' % rid)
                        warn('Solver status is "time_limit" with reaction %s' % rid)
                    elif 'feasibility' in str(w) and prev_sol_bin[idx] == 1:
                        print('The solver has encountered an infeasible optimization with reaction %s. '
                              'The model may be infeasible when this reaction is blocked. If this happens '
                              'frequently, the model may contain many blocked reactions, or there may be a problem '
                              'with the starting solution, or the tolerance parameters.' % rid)
                        warn('Solver status is "infeasible" when reaction %s is blocked' % rid)
                    elif 'feasibility' in str(w) and prev_sol_bin[idx] == 0:
                        print('The solver has encountered an infeasible optimization with reaction %s. '
                              'The model may be infeasible when this reaction is irreversible. If this happens '
                              'frequently, there may be a problem with the starting solution, or the tolerance '
                              'parameters.' % rid)
                        warn('Solver status is "infeasible" when reaction %s is irreversible' % rid)
                    else:
                        print('An unexpected error has occured during the solver call with reaction %s.' % rid)
                        warn(str(w))
    solution = RxnEnumSolution(all_solutions, unique_solutions, all_solutions_binary, unique_solutions_binary,
                               all_reactions, unique_reactions, prev_sol.objective_value)
    return solution


def main():
    """
    This function is called when you run this script from the commandline.
    It performs the reaction-enumeration algorithm on a specified list of reactions
    Use --help to see commandline parameters
    """
    description = 'Performs the reaction-enumeration algorithm on a specified list of reactions'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, matlab, or json format')
    parser.add_argument('-l', '--reaction_list', default=None, help='csv list of reactions to enumerate - if empty, '
                                                                    'will use all reactions in the model')
    parser.add_argument('--range', default='_',
                        help='range of reactions to use from the list, in the format "integer_integer", 0-indexed')
    parser.add_argument('-r', '--reaction_weights', default=None,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-p', '--prev_sol', default=None, help='initial imat solution in .txt format')
    parser.add_argument('-e', '--epsilon', type=float, default=DEFAULT_VALUES['epsilon'],
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=DEFAULT_VALUES['threshold'],
                        help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=DEFAULT_VALUES['timelimit'], help='Solver time limit')
    parser.add_argument('--tol', type=float, default=DEFAULT_VALUES['tolerance'], help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=DEFAULT_VALUES['mipgap'], help='Solver MIP gap tolerance')
    parser.add_argument('--obj_tol', type=float, default=DEFAULT_VALUES['obj_tol'],
                        help='objective value tolerance, as a fraction of the original value')
    parser.add_argument('-o', '--output', default='rxn_enum', help='Path of output files, without format')
    parser.add_argument('--save', action='store_true', help='Use this flag to save each solution individually')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)

    reaction_weights = {}
    if args.reaction_weights is not None:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    if args.reaction_list is not None:
        df = pd.read_csv(args.reaction_list, header=None)
        reactions = [x for x in df.unstack().values]
    else:
        reactions = [r.id for r in model.reactions]

    rxn_range = args.range.split('_')
    if rxn_range[0] == '':
        start = 0
    else:
        start = int(rxn_range[0])
    if rxn_range[1] == '':
        rxn_list = reactions[start:]
    elif int(rxn_range[1]) >= len(reactions):
        rxn_list = reactions[start:]
    else:
        rxn_list = reactions[start:int(rxn_range[1])]
    prev_sol, _ = read_prev_sol(prev_sol_arg=args.prev_sol, model=model, rw=reaction_weights, eps=args.epsilon,
                                thr=args.threshold)

    solution = rxn_enum(model=model, rxn_list=rxn_list, prev_sol=prev_sol, reaction_weights=reaction_weights,
                        eps=args.epsilon, thr=args.threshold, obj_tol=args.obj_tol, out_path=args.output,
                        save=args.save)
    uniques = pd.DataFrame(solution.unique_binary)
    uniques.columns = [r.id for r in model.reactions]
    uniques.to_csv(args.output + '_solutions.csv')
    return True


if __name__ == '__main__':
    main()

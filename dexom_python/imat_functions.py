import argparse
import time

import cobra
import cobra.util.array
import optlang
from symengine import Add, sympify
from numpy import abs, max
from warnings import catch_warnings, filterwarnings, resetwarnings
from cobra.exceptions import OptimizationError
from dexom_python.model_functions import read_model, check_model_options, load_reaction_weights, check_threshold_tolerance, check_constraint_values, check_model_primals
from dexom_python.result_functions import write_solution
from dexom_python.default_parameter_values import DEFAULT_VALUES


class ImatException(Exception):
    pass


def create_optimality_constraint(model, reaction_weights, prev_sol, obj_tol=DEFAULT_VALUES['obj_tol'], name='optimality', full=False):
    """
    Creates the optimality constraint  based on the imat objective function
    This constraint conserves the optimal objective value of the previous solution
    For detailed explanation of parameters see documentation of imat function.

    Parameters
    ----------
    model: cobra.Model
    reaction_weights: dict
    prev_sol: cobra.Solution or float
        either the previous iMAT solution or the objective value of that solution
    obj_tol: float
        variance allowed in the objective_value of the solution
    name: string
    full: bool

    Returns
    -------
    optlang Constraint object (dependent on current solver)

    """
    if isinstance(prev_sol, cobra.core.solution.Solution):
        lower_opt = prev_sol.objective_value - prev_sol.objective_value * obj_tol
    else:
        lower_opt = prev_sol - prev_sol * obj_tol
    variables = []
    weights = []
    try:
        for rid, weight in reaction_weights.items():
            if weight > 0:
                variables.append(model.solver.variables['x_' + rid])
                weights.append(weight)
            elif weight < 0:
                variables.append(sympify('1') - model.solver.variables['x_' + rid])
                weights.append(abs(weight))
    except KeyError as e:
        raise Exception('Searching for the reaction_weights in the model raised a KeyError, verify that all indexes '
                        'from reaction_weights are present in the model and spelled correctly') from e
    opt_const = model.solver.interface.Constraint(Add(*[x * w for x, w in zip(variables, weights)]), lb=lower_opt, name=name)
    return opt_const


def create_full_variable_single(model, rid, reaction_weights, epsilon, threshold):
    # the x_rid variables represent a binary condition of flux activation
    if 'x_' + rid not in model.solver.variables:
        rxn = model.reactions.get_by_id(rid)
        xtot = model.solver.interface.Variable('x_%s' % rid, type='binary')
        xf = model.solver.interface.Variable('xf_%s' % rid, type='binary')
        xr = model.solver.interface.Variable('xr_%s' % rid, type='binary')
        model.solver.add(xtot)
        model.solver.add(xf)
        model.solver.add(xr)
        xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name='x_%s_def' % rid)
        temp = threshold
        if reaction_weights.get(rid, 0.) > 0.:
            temp = epsilon
        up = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               - xf * temp - xr * rxn.lower_bound, lb=0., name='%s_lower' % rid)
        lo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               + xr * temp - xf * rxn.upper_bound, ub=0., name='%s_upper' % rid)
        model.solver.add(up)
        model.solver.add(lo)
        model.solver.add(xtot_def)
    return model


def create_new_partial_variable_single(model, rid, epsilon, threshold, pos):
    # the variable definition is more precise than in the original iMAT implementation
    # this is done in order to avoid problems with enumeration methods, and doesn't affect the results of iMAT
    try:
        integ = model.solver.configuration.tolerances.integrality
        feasib = model.solver.configuration.tolerances.feasibility
    except:
        print('This implementation has been written with the cplex solver. It should work with other solvers, but may behave differently.')
        integ = feasib = model.tolerance
    rllimit = integ * max([r.bounds for r in model.reactions]) + feasib
    if pos and 'x_' + rid not in model.solver.variables:
        rxn = model.reactions.get_by_id(rid)
        xtot = model.solver.interface.Variable('x_%s' % rid, type='binary')
        xf = model.solver.interface.Variable('xf_%s' % rid, type='binary')
        xr = model.solver.interface.Variable('xr_%s' % rid, type='binary')
        model.solver.add(xtot)
        model.solver.add(xf)
        model.solver.add(xr)
        xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name='x_%s_def' % rid)
        model.solver.add(xtot_def)
        lo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               - xf * epsilon + xf * rxn.lower_bound, lb=rxn.lower_bound, name='%s_lower' % rid)
        up = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               + xr * epsilon + xr * rxn.upper_bound, ub=rxn.upper_bound, name='%s_upper' % rid)
        lofo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                                 - xf * rxn.upper_bound, ub=epsilon-rllimit, name='_%s_forcelower' % rid)
        upfo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                                 - xr * rxn.lower_bound, lb=-epsilon+rllimit, name='_%s_forceupper' % rid)
        model.solver.add(up)
        model.solver.add(lo)
        model.solver.add(lofo)
        model.solver.add(upfo)
    elif not pos and 'x_' + rid not in model.solver.variables:
        rxn = model.reactions.get_by_id(rid)
        xtot = model.solver.interface.Variable('x_%s' % rid, type='binary')
        xf = model.solver.interface.Variable('xf_%s' % rid, type='binary')
        xr = model.solver.interface.Variable('xr_%s' % rid, type='binary')
        model.solver.add(xtot)
        model.solver.add(xf)
        model.solver.add(xr)
        xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name='x_%s_def' % rid)
        model.solver.add(xtot_def)
        lo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               - xtot * rxn.lower_bound, lb=-threshold, name='%s_lower' % rid)
        up = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               - xtot * rxn.upper_bound, ub=threshold, name='%s_upper' % rid)
        lofo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                                 + xr * rxn.upper_bound + xr * (threshold), ub=rxn.upper_bound, name='_%s_forcelower' % rid)
        upfo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                                 + xf * rxn.lower_bound - xf * (threshold), lb=rxn.lower_bound, name='_%s_forceupper' % rid)
        model.solver.add(up)
        model.solver.add(lo)
        model.solver.add(lofo)
        model.solver.add(upfo)
    return model


def create_full_variables(model, reaction_weights, epsilon, threshold):
    """
    Creates binary indicator variables in the model for every reaction.
    """
    for rxn in model.reactions:
        model = create_full_variable_single(model=model, rid=rxn.id, reaction_weights=reaction_weights,
                                            epsilon=epsilon, threshold=threshold)
    return model


def create_new_partial_variables(model, reaction_weights, epsilon, threshold):
    """
    Creates binary indicator variables in the model for reactions with nonzero weight.
    """
    for rxn in model.reactions:
        if rxn.id in reaction_weights.keys():
            weight = reaction_weights[rxn.id]
            if weight != 0.:
                pos = weight>0
                model = create_new_partial_variable_single(model=model, rid=rxn.id, epsilon=epsilon,
                                                           threshold=threshold, pos=pos)
    return model


def _imat_call_model_optimizer(model):
    t1 = time.perf_counter()
    with catch_warnings():
        filterwarnings('error')
        try:
            # with model:
            solution = model.optimize()
            t2 = time.perf_counter()
            print('%.2fs during optimize call' % (t2 - t1))
            if isinstance(model.solver, optlang.glpk_interface.Model):
                # during reaction-enumeration, GLPK sometimes returns invalid solutions
                check_constraint_values(model)
            return solution
        except UserWarning as w:
            resetwarnings()
            if 'time_limit' in str(w):
                print('The solver has reached the timelimit. This can happen if there are too many constraints on '
                      'the model, or if some of the following parameters have too low values: epsilon, threshold, '
                      'feasibility tolerance, MIP gap tolerance.')
                # warn('Solver status is "time_limit"')
                raise ImatException('Solver status is "time_limit", timelimit error')
            elif 'infeasible' in str(w):
                print('The solver has encountered an infeasible optimization. This can happen if there are too '
                      'many constraints on the model, or if some of the following parameters have too low values: '
                      'epsilon, threshold, feasibility tolerance, MIP gap tolerance.')
                # warn('Solver status is "infeasible"')
                raise ImatException('Solver status is "infeasible", feasibility error')
            else:
                print('An unexpected error has occured during the solver call')
                # warn(w)
                raise ImatException(str(w))
        except OptimizationError as e:
            resetwarnings()
            print('An unexpected error has occured during the solver call.')
            raise ImatException(str(e))


def parsimonious_imat(model, reaction_weights=None, prev_sol=None, obj_tol=0., epsilon=DEFAULT_VALUES['epsilon'],
                      threshold=DEFAULT_VALUES['threshold'], full=False):
    """
    This function applies parsimonious iMAT:
    Parameters
    ----------
    model
    reaction_weights
    prev_sol
    obj_tol
    epsilon
    threshold
    full

    Returns
    -------
    solution: a cobrapy Solution
    """
    epsilon, threshold = check_threshold_tolerance(model=model, epsilon=epsilon, threshold=threshold)
    if reaction_weights is None:
        reaction_weights = {}
    if prev_sol is None:
        prev_sol = imat(model=model, reaction_weights=reaction_weights, epsilon=epsilon, threshold=threshold, full=full)

    opt_const = create_optimality_constraint(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol,
                                             obj_tol=obj_tol)
    model.solver.add(opt_const)

    reaction_variables = []
    for rxn in model.reactions:
        reaction_variables.extend([rxn.forward_variable, rxn.reverse_variable])
    objective = model.solver.interface.Objective(Add(*reaction_variables), direction='min')
    model.objective = objective

    solution = model.optimize()

    return solution


def imat(model, reaction_weights=None, epsilon=DEFAULT_VALUES['epsilon'], threshold=DEFAULT_VALUES['threshold'],
         full=False):
    """
    Modified version of the integrative Metabolic Analysis Tool with reaction weights

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    reaction_weights: dict
        keys are reaction IDs, values are weights
    epsilon: float
        activation threshold for highly expressed reactions
    threshold: float
        activation threshold for all reactions
    full: bool
        if True, create variables for all reactions. if False, only for reactions with non-zero weights

    Returns
    -------
    solution: cobra.Solution
    """
    epsilon, threshold = check_threshold_tolerance(model=model, epsilon=epsilon, threshold=threshold)
    if reaction_weights is None:
        reaction_weights = {}
    y_variables = list()
    x_variables = list()
    y_weights = list()
    x_weights = list()
    t0 = time.perf_counter()
    try:
        if full:  # for the full_imat implementation
            model = create_full_variables(model, reaction_weights, epsilon, threshold)
        else:
            model = create_new_partial_variables(model, reaction_weights, epsilon, threshold)
        for rid, weight in reaction_weights.items():
            if rid not in model.reactions:
                UserWarning(f'reactions {rid} is not present in the model, will be ignored')
            elif weight > 0:
                y_pos = model.solver.variables['xf_' + rid]
                y_neg = model.solver.variables['xr_' + rid]
                y_variables.append([y_neg, y_pos])
                y_weights.append(weight)
            elif weight < 0 :
                x = sympify('1') - model.solver.variables['x_' + rid]
                # x = model.solver.variables['x_' + rid]
                x_variables.append(x)
                x_weights.append(abs(weight))
        rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
        rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
        objective = model.solver.interface.Objective(Add(*rh_objective) + Add(*rl_objective), direction='max')
        model.objective = objective
        t1 = time.perf_counter()
        print('%.2fs before optimize call' % (t1 - t0))
        solution = _imat_call_model_optimizer(model)
        var_primals = model.solver.primal_values
        rh = 0
        rl = 0
        rhtot = 0
        rltot = 0
        for rid, weight in reaction_weights.items():
            if rid in model.reactions:
                if weight > 0:
                    x = var_primals['x_' + rid]
                    rhtot += 1
                    rh += int(x)
                elif weight < 0:
                    x = var_primals['x_' + rid]
                    rltot += 1
                    rl += 1 - int(x)
        print('Objective value: ', solution.objective_value)
        if rhtot + rltot == 0 :
            print('No valid reaction-weights in model, optimal objective is zero.')
        else:
            print(f'Fraction of optimal objective value: {100*(rh+rl)/(rhtot+rltot)}%')
        print(f'RH: {rh} out of {rhtot} reactions active')
        print(f'RL: {rl} out of {rltot} reactions inactive')
        return solution
    finally:
        pass


def _main():
    """
    This function is called when you run this script from the commandline.
    It performs the modified iMAT algorithm with reaction weights.
    Use --help to see commandline parameters
    """
    description = 'Performs the modified iMAT algorithm with reaction weights'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default=argparse.SUPPRESS,
                        help='Metabolic model in sbml, json, or matlab format')
    parser.add_argument('-r', '--reaction_weights', default=argparse.SUPPRESS,
                        help='Reaction weights in csv format with column names: (reactions, weights)')
    parser.add_argument('-e', '--epsilon', type=float, default=DEFAULT_VALUES['epsilon'],
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=DEFAULT_VALUES['threshold'],
                        help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=DEFAULT_VALUES['timelimit'], help='Solver time limit')
    parser.add_argument('--tol', type=float, default=DEFAULT_VALUES['tolerance'], help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=DEFAULT_VALUES['mipgap'], help='Solver MIP gap tolerance')
    parser.add_argument('-o', '--output', default='imat_solution', help='Path of the output file, without format')
    parser.add_argument('-p', '--parsimony', action='store_true', help='Use this flag to perform parsimonious imat')
    args = parser.parse_args()
    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, tolerance=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)
    solution = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold)
    if args.parsimony:
        objective_value = solution.objective_value
        solution = parsimonious_imat(model, reaction_weights, prev_sol=solution, obj_tol=0.,
                                     epsilon=args.epsilon, threshold=args.threshold)
        solution.objective_value = objective_value
    write_solution(model, solution, args.threshold, args.output+'.csv')
    return True


if __name__ == '__main__':
    _main()

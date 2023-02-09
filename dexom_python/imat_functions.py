import six
import argparse
import time
from symengine import Add, sympify
from numpy import abs
from warnings import catch_warnings, filterwarnings, resetwarnings
from cobra.exceptions import OptimizationError
from dexom_python.model_functions import read_model, check_model_options, load_reaction_weights, DEFAULT_VALUES, check_threshold_tolerance
from dexom_python.result_functions import write_solution


class ImatException(Exception):
    pass


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
        if rid in reaction_weights:
            if reaction_weights[rid] > 0.:
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
        up = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               - xf * epsilon - xr * rxn.lower_bound, lb=0., name='%s_lower' % rid)
        lo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               + xr * epsilon - xf * rxn.upper_bound, ub=0., name='%s_upper' % rid)
        model.solver.add(up)
        model.solver.add(lo)
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
        up = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               - xf * threshold - xr * rxn.lower_bound, lb=0., name='%s_lower' % rid)
        lo = model.solver.interface.Constraint(rxn.forward_variable - rxn.reverse_variable
                                               + xr * threshold - xf * rxn.upper_bound, ub=0., name='%s_upper' % rid)
        model.solver.add(up)
        model.solver.add(lo)
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
            if weight > 0:
                model = create_new_partial_variable_single(model=model, rid=rxn.id, epsilon=epsilon,
                                                           threshold=threshold, pos=True)
            elif weight < 0:  # the rl_rid variables represent the lowly expressed reactions
                model = create_new_partial_variable_single(model=model, rid=rxn.id, epsilon=epsilon,
                                                           threshold=threshold, pos=False)
    return model


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
    check_threshold_tolerance(model=model, epsilon=epsilon, threshold=threshold)
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
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0 and rid in model.reactions:
                y_pos = model.solver.variables['xf_' + rid]
                y_neg = model.solver.variables['xr_' + rid]
                y_variables.append([y_neg, y_pos])
                y_weights.append(weight)
            elif weight < 0 and rid in model.reactions:
                x = sympify('1') - model.solver.variables['x_' + rid]
                x_variables.append(x)
                x_weights.append(abs(weight))
        rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
        rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
        objective = model.solver.interface.Objective(Add(*rh_objective) + Add(*rl_objective), direction='max')
        model.objective = objective
        t1 = time.perf_counter()
        print('%.2fs before optimize call' % (t1 - t0))
        with catch_warnings():
            filterwarnings('error')
            try:
                # with model:
                solution = model.optimize()
                t2 = time.perf_counter()
                print('%.2fs during optimize call' % (t2-t1))
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
    finally:
        pass


def _main():
    """
    This function is called when you run this script from the commandline.
    It performs the modified iMAT algorithm with reaction weights.
    Use --help to see commandline parameters
    """
    description = 'Performs the modified iMAT algorithm with reaction weights'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, json, or matlab format')
    parser.add_argument('-r', '--reaction_weights', default={},
                        help='Reaction weights in csv format with column names: (reactions, weights)')
    parser.add_argument('-e', '--epsilon', type=float, default=DEFAULT_VALUES['epsilon'],
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=DEFAULT_VALUES['threshold'],
                        help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=DEFAULT_VALUES['timelimit'], help='Solver time limit')
    parser.add_argument('--tol', type=float, default=DEFAULT_VALUES['tolerance'], help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=DEFAULT_VALUES['mipgap'], help='Solver MIP gap tolerance')
    parser.add_argument('-o', '--output', default='imat_solution', help='Path of the output file, without format')
    args = parser.parse_args()
    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)
    solution = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold)
    write_solution(model, solution, args.threshold, args.output+'.csv')
    return True


if __name__ == '__main__':
    _main()

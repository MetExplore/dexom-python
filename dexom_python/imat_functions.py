import six
from symengine import Add, sympify
from numpy import abs
import argparse
import time
from dexom_python.model_functions import read_model, check_model_options, load_reaction_weights
from dexom_python.result_functions import write_solution


def create_full_variables(model, reaction_weights, epsilon, threshold):
    # the x_rid variables represent a binary condition of flux activation
    for rxn in model.reactions:
        if 'x_' + rxn.id not in model.solver.variables:
            rid = rxn.id
            xtot = model.solver.interface.Variable('x_%s' % rid, type='binary')
            xf = model.solver.interface.Variable('xf_%s' % rid, type='binary')
            xr = model.solver.interface.Variable('xr_%s' % rid, type='binary')
            model.solver.add(xtot)
            model.solver.add(xf)
            model.solver.add(xr)
            xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name='x_%s_def' % rid)
            xf_upper = model.solver.interface.Constraint(
                rxn.forward_variable - rxn.upper_bound * xf, ub=0., name='xr_%s_upper' % rid)
            xr_upper = model.solver.interface.Constraint(
                rxn.reverse_variable + rxn.lower_bound * xr, ub=0., name='xf_%s_upper' % rid)
            temp = threshold
            if rid in reaction_weights:
                if reaction_weights[rid] > 0.:
                    temp = epsilon
            xf_lower = model.solver.interface.Constraint(
                rxn.forward_variable - temp * xf, lb=0., name='xf_%s_lower' % rid)
            xr_lower = model.solver.interface.Constraint(
                rxn.reverse_variable - temp * xr, lb=0., name='xr_%s_lower' % rid)
            model.solver.add(xtot_def)
            model.solver.add(xf_upper)
            model.solver.add(xr_upper)
            model.solver.add(xf_lower)
            model.solver.add(xr_lower)
    return model


def create_new_partial_variables(model, reaction_weights, epsilon, threshold):
    # the variable definition is more precise than in the original iMAT implementation
    # this is done in order to avoid problems with enumeration methods, and doesn't affect the results of iMAT
    for rid, weight in six.iteritems(reaction_weights):
        if weight > 0 and rid in model.reactions:
            if 'rh_' + rid + '_pos' not in model.solver.variables:
                rxn = model.reactions.get_by_id(rid)
                xtot = model.solver.interface.Variable('x_%s' % rid, type='binary')
                xf = model.solver.interface.Variable('rh_%s_pos' % rid, type='binary')
                xr = model.solver.interface.Variable('rh_%s_neg' % rid, type='binary')
                model.solver.add(xtot)
                model.solver.add(xf)
                model.solver.add(xr)
                xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name='x_%s_def' % rid)
                xf_upper = model.solver.interface.Constraint(
                    rxn.forward_variable - rxn.upper_bound * xf, ub=0., name='xr_%s_upper' % rid)
                xr_upper = model.solver.interface.Constraint(
                    rxn.reverse_variable + rxn.lower_bound * xr, ub=0., name='xf_%s_upper' % rid)
                xf_lower = model.solver.interface.Constraint(
                    rxn.forward_variable - epsilon * xf, lb=0., name='xf_%s_lower' % rid)
                xr_lower = model.solver.interface.Constraint(
                    rxn.reverse_variable - epsilon * xr, lb=0., name='xr_%s_lower' % rid)
                model.solver.add(xtot_def)
                model.solver.add(xf_upper)
                model.solver.add(xr_upper)
                model.solver.add(xf_lower)
                model.solver.add(xr_lower)
        elif weight < 0 and rid in model.reactions:  # the rl_rid variables represent the lowly expressed reactions
            if 'rl_' + rid not in model.solver.variables:
                rxn = model.reactions.get_by_id(rid)
                xtot = model.solver.interface.Variable('rl_%s' % rid, type='binary')
                xf = model.solver.interface.Variable('xf_%s' % rid, type='binary')
                xr = model.solver.interface.Variable('xr_%s' % rid, type='binary')
                model.solver.add(xtot)
                model.solver.add(xf)
                model.solver.add(xr)
                xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name='x_%s_def' % rid)
                xf_upper = model.solver.interface.Constraint(
                    rxn.forward_variable - rxn.upper_bound * xf, ub=0., name='xr_%s_upper' % rid)
                xr_upper = model.solver.interface.Constraint(
                    rxn.reverse_variable + rxn.lower_bound * xr, ub=0., name='xf_%s_upper' % rid)
                xf_lower = model.solver.interface.Constraint(
                    rxn.forward_variable - threshold * xf, lb=0., name='xf_%s_lower' % rid)
                xr_lower = model.solver.interface.Constraint(
                    rxn.reverse_variable - threshold * xr, lb=0., name='xr_%s_lower' % rid)
                model.solver.add(xtot_def)
                model.solver.add(xf_upper)
                model.solver.add(xr_upper)
                model.solver.add(xf_lower)
                model.solver.add(xr_lower)
    return model


def imat(model, reaction_weights=None, epsilon=1e-2, threshold=1e-5, full=False):
    """
    Integrative Metabolic Analysis Tool

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    reaction_weights: dict
        keys are reaction ids, values are int weights
    epsilon: float
        activation threshold for highly expressed reactions
    threshold: float
        activation threshold for all reactions
    full: bool
        if True, apply constraints on all reactions. if False, only on reactions with non-zero weights
    Returns
    -------
    solution: cobra.Solution
    """
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
        else:
            model = create_new_partial_variables(model, reaction_weights, epsilon, threshold)
            for rid, weight in six.iteritems(reaction_weights):
                if weight > 0 and rid in model.reactions:
                    y_neg = model.solver.variables['rh_' + rid + '_neg']
                    y_pos = model.solver.variables['rh_' + rid + '_pos']
                    y_variables.append([y_neg, y_pos])
                    y_weights.append(weight)
                elif weight < 0 and rid in model.reactions:
                    x = sympify('1') - model.solver.variables['rl_' + rid]
                    x_variables.append(x)
                    x_weights.append(abs(weight))
        rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
        rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
        objective = model.solver.interface.Objective(Add(*rh_objective) + Add(*rl_objective), direction='max')
        model.objective = objective
        t1 = time.perf_counter()
        with model:
            solution = model.optimize()
            t2 = time.perf_counter()
            print('%.2fs before optimize call' % (t1-t0))
            print('%.2fs during optimize call' % (t2-t1))
            return solution
    finally:
        pass


if __name__ == '__main__':
    description = 'Performs the iMAT algorithm'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Metabolic model in sbml, json, or matlab format')
    parser.add_argument('-r', '--reaction_weights', default={},
                        help='Reaction weights in csv format with column names: (reactions, weights)')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-4,
                        help='Activation threshold for highly expressed reactions')
    parser.add_argument('--threshold', type=float, default=1e-8, help='Activation threshold for all reactions')
    parser.add_argument('-t', '--timelimit', type=int, default=None, help='Solver time limit')
    parser.add_argument('--tol', type=float, default=1e-8, help='Solver feasibility tolerance')
    parser.add_argument('--mipgap', type=float, default=1e-6, help='Solver MIP gap tolerance')
    parser.add_argument('-o', '--output', default='imat_solution', help='Path of the output file, without format')
    args = parser.parse_args()

    model = read_model(args.model)
    check_model_options(model, timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)
    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)
    solution = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold)
    write_solution(model, solution, args.threshold, args.output+'.csv')

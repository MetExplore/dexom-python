
import six
from symengine import Add, sympify
from numpy import abs
import argparse
from pathlib import Path
import time
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from model_functions import load_reaction_weights
from result_functions import write_solution


def create_full_variables(model, reaction_weights, epsilon, threshold):

    # the x_rid variables represent a binary condition of flux activation
    for rxn in model.reactions:
        if "x_" + rxn.id not in model.solver.variables:
            rid = rxn.id
            xtot = model.solver.interface.Variable("x_%s" % rid, type="binary")
            xf = model.solver.interface.Variable("xf_%s" % rid, type="binary")
            xr = model.solver.interface.Variable("xr_%s" % rid, type="binary")
            model.solver.add(xtot)
            model.solver.add(xf)
            model.solver.add(xr)
            xtot_def = model.solver.interface.Constraint(xtot - xf - xr, lb=0., ub=0., name="x_%s_def" % rid)
            xf_upper = model.solver.interface.Constraint(
                rxn.forward_variable - rxn.upper_bound * xf, ub=0., name="xr_%s_upper" % rid)
            xr_upper = model.solver.interface.Constraint(
                rxn.reverse_variable + rxn.lower_bound * xr, ub=0., name="xf_%s_upper" % rid)
            temp = threshold
            if rid in reaction_weights:
                if reaction_weights[rid] > 0.:
                    temp = epsilon
            xf_lower = model.solver.interface.Constraint(
                rxn.forward_variable - temp * xf, lb=0., name="xf_%s_lower" % rid)
            xr_lower = model.solver.interface.Constraint(
                rxn.reverse_variable - temp * xr, lb=0., name="xr_%s_lower" % rid)
            model.solver.add(xtot_def)
            model.solver.add(xf_upper)
            model.solver.add(xr_upper)
            model.solver.add(xf_lower)
            model.solver.add(xr_lower)

    return model


def create_partial_variables(model, reaction_weights, epsilon):

    for rid, weight in six.iteritems(reaction_weights):
        if weight > 0:  # the rh_rid variables represent the highly expressed reactions
            if "rh_" + rid + "_pos" not in model.solver.variables:
                reaction = model.reactions.get_by_id(rid)
                y_pos = model.solver.interface.Variable("rh_%s_pos" % rid, type="binary")
                y_neg = model.solver.interface.Variable("rh_%s_neg" % rid, type="binary")
                pos_constraint = model.solver.interface.Constraint(
                    reaction.flux_expression + y_pos * (reaction.lower_bound - epsilon),
                    lb=reaction.lower_bound, name="rh_%s_pos_bound" % rid)
                neg_constraint = model.solver.interface.Constraint(
                    reaction.flux_expression + y_neg * (reaction.upper_bound + epsilon),
                    ub=reaction.upper_bound, name="rh_%s_neg_bound" % rid)
                model.solver.add(y_pos)
                model.solver.add(y_neg)
                model.solver.add(pos_constraint)
                model.solver.add(neg_constraint)

        elif weight < 0:  # the rl_rid variables represent the lowly expressed reactions
            if "rl_" + rid not in model.solver.variables:
                reaction = model.reactions.get_by_id(rid)
                x = model.solver.interface.Variable("rl_%s" % rid, type="binary")
                pos_constraint = model.solver.interface.Constraint(
                    (1 - x) * reaction.upper_bound - reaction.flux_expression,
                    lb=0, name="rl_%s_upper" % rid)
                neg_constraint = model.solver.interface.Constraint(
                    (1 - x) * reaction.lower_bound - reaction.flux_expression,
                    ub=0, name="rl_%s_lower" % rid)
                model.solver.add(x)
                model.solver.add(pos_constraint)
                model.solver.add(neg_constraint)

    return model


def imat(model, reaction_weights={}, epsilon=1e-2, threshold=1e-5, timelimit=None, feasibility=1e-6, mipgaptol=1e-3,
         full=False):
    """
    Integrative Metabolic Analysis Tool

    Parameters
    ----------
    model: cobra.Model
        A constraint-based model
    reaction_weights: dict
        keys are reaction ids, values are int weights
    epsilon: float
        activation threshold for highly expressed reactions
    threshold: float
        activation threshold for all reactions
    timelimit: int
        time limit (in seconds) for the model.optimize() call
    feasibility: float
        feasibility tolerance of the solver
    mipgaptol: float
        MIP Gap tolerance of the solver
    full: bool
        if True, apply constraints on all reactions. if False, only on reactions with non-zero weights
    """
    try:
        model.solver = 'cplex'
    except:
        print("cplex is not available or not properly installed")

    y_variables = list()
    x_variables = list()
    y_weights = list()
    x_weights = list()
    t0 = time.perf_counter()
    try:
        if full:  # for the full_imat implementation
            model = create_full_variables(model, reaction_weights, epsilon, threshold)
            for rid, weight in six.iteritems(reaction_weights):
                if weight > 0:
                    y_pos = model.solver.variables["xf_" + rid]
                    y_neg = model.solver.variables["xr_" + rid]
                    y_variables.append([y_neg, y_pos])
                    y_weights.append(weight)
                elif weight < 0:
                    x = sympify("1") - model.solver.variables["x_" + rid]
                    x_variables.append(x)
                    x_weights.append(abs(weight))

        else:  # for the driven-based imat implementation
            model = create_partial_variables(model, reaction_weights, epsilon)
            for rid, weight in six.iteritems(reaction_weights):
                if weight > 0:
                    y_neg = model.solver.variables["rh_" + rid + "_neg"]
                    y_pos = model.solver.variables["rh_" + rid + "_pos"]
                    y_variables.append([y_neg, y_pos])
                    y_weights.append(weight)
                elif weight < 0:
                    x = model.solver.variables["rl_" + rid]
                    x_variables.append(x)
                    x_weights.append(abs(weight))

        rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
        rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
        objective = model.solver.interface.Objective(Add(*rh_objective) + Add(*rl_objective), direction="max")
        model.objective = objective
        model.solver.configuration.timeout = timelimit
        model.tolerance = feasibility
        model.solver.problem.parameters.mip.tolerances.mipgap.set(mipgaptol)
        model.solver.configuration.presolve = True
        t1 = time.perf_counter()
        with model:
            solution = model.optimize()
            t2 = time.perf_counter()
            print("iMAT:", t1-t0, "s spent before optimize call")
            print("iMAT:", t2-t1, "s spent on optimize call")
            return solution
    finally:
        pass


if __name__ == "__main__":
    description = "Performs the iMAT algorithm"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in sbml, json, or matlab format")
    parser.add_argument("-r", "--reaction_weights", default={},
                        help="Reaction weights in csv format with column names: (reactions, weights)")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-2,
                        help="Activation threshold for highly expressed reactions")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Activation threshold for all reactions")
    parser.add_argument("-t", "--timelimit", type=int, default=None, help="Solver time limit")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver feasibility tolerance")
    parser.add_argument("--mipgap", type=float, default=1e-3, help="Solver MIP gap tolerance")
    parser.add_argument("-o", "--output", default="imat_solution", help="Path of the output file, without format")

    args = parser.parse_args()

    fileformat = Path(args.model).suffix
    if fileformat == ".sbml" or fileformat == ".xml":
        model = read_sbml_model(args.model)
    elif fileformat == '.json':
        model = load_json_model(args.model)
    elif fileformat == ".mat":
        model = load_matlab_model(args.model)
    else:
        print("Only SBML, JSON, and Matlab formats are supported for the models")
        model = None

    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    solution = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold, timelimit=args.timelimit,
                    feasibility=args.tol, mipgaptol=args.mipgap)

    write_solution(solution, args.threshold, args.output+".csv")

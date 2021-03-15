
from cobra import Model
import six
from sympy import Add
from numpy import abs
import argparse
from sympy import sympify


def imat(model, reaction_weights=None, threshold=1, *args, **kwargs):
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
    """

    assert isinstance(model, Model)
    assert isinstance(reaction_weights, dict)

    y_variables = list()
    x_variables = list()
    y_weights = list()
    x_weights = list()

    try:
        # the x_rid variables represent a binary condition of flux activation
        for rxn in model.reactions:
            if "x_"+rxn.id not in model.solver.variables:
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
                xf_lower = model.solver.interface.Constraint(
                    rxn.forward_variable - threshold * xf, lb=0., name="xf_%s_lower" % rid)
                xr_lower = model.solver.interface.Constraint(
                    rxn.reverse_variable - threshold * xr, lb=0., name="xr_%s_lower" % rid)
                model.solver.add(xtot_def)
                model.solver.add(xf_upper)
                model.solver.add(xr_upper)
                model.solver.add(xf_lower)
                model.solver.add(xr_lower)

        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:  # the rh_rid variables represent the highly expressed reactions
                y_pos = model.solver.variables["xf_"+rid]
                y_neg = model.solver.variables["xr_"+rid]
                y_variables.append([y_neg, y_pos])
                y_weights.append(weight)

            elif weight < 0:  # the rl_rid variables represent the lowly expressed reactions
                x = sympify("1") - model.solver.variables["x_"+rid]
                x_variables.append(x)
                x_weights.append(abs(weight))

        rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
        rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
        objective = model.solver.interface.Objective(Add(*rh_objective) + Add(*rl_objective), direction="max")

        with model:
            model.objective = objective
            solution = model.optimize()
            return solution
    finally:
        pass


if __name__ == "__main__":
    description = "Performs the iMAT algorithm"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in python format")
    parser.add_argument("-r", "--reactionweights", default=None, help="Reaction weights in dict format")
    parser.add_argument("-e", "--epsilon", default=1., help="Activation threshold for highly expressed reactions")
    parser.add_argument("-t", "--threshold", default=1e-1, help="Activation threshold for all reactions")

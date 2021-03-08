
from cobra import Model
import six
from sympy import Add


def imat(model, reaction_weights=None, epsilon=0.1, threshold = 1e-3, *args, **kwargs):
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
    constraints = list()

    y_weights = list()
    x_weights = list()

    try:
        # the x_rid variables represent a binary condition of flux activation
        for rxn in model.reactions:
            if "x_"+rxn.id not in model.solver.variables:
                rid = rxn.id
                bin_var = model.solver.interface.Variable("x_%s" % rid, type="binary")
                bin_upp = model.solver.interface.Constraint(
                    rxn.lower_bound * bin_var - rxn.flux_expression, ub=threshold, name="x_%s_upper" % rid)
                bin_low = model.solver.interface.Constraint(
                    rxn.upper_bound * bin_var - rxn.flux_expression, lb=-threshold, name="x_%s_lower" % rid)
                model.solver.add(bin_var)
                model.solver.add(bin_upp)
                model.solver.add(bin_low)

        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:  # the rh_rid variables represent the highly expressed reactions
                reaction = model.reactions.get_by_id(rid)
                y_pos = model.solver.interface.Variable("rh_%s_pos" % rid, type="binary")
                y_neg = model.solver.interface.Variable("rh_%s_neg" % rid, type="binary")

                y_variables.append([y_neg, y_pos])

                pos_constraint = model.solver.interface.Constraint(
                    reaction.flux_expression + y_pos * (reaction.lower_bound - epsilon),
                    lb=reaction.lower_bound, name="rh_%s_pos_bound" % rid)

                neg_constraint = model.solver.interface.Constraint(
                    reaction.flux_expression + y_neg * (reaction.upper_bound + epsilon),
                    ub=reaction.upper_bound, name="rh_%s_neg_bound" % rid)

                constraints.extend([pos_constraint, neg_constraint])

                y_weights.append(weight)

            elif weight < 0:  # the rl_rid variables represent the lowly expressed reactions
                reaction = model.reactions.get_by_id(rid)
                x = model.solver.interface.Variable("rl_%s" % rid, type="binary")
                x_variables.append(x)

                pos_constraint = model.solver.interface.Constraint(
                    (1 - x) * reaction.upper_bound - reaction.flux_expression,
                    lb=0, name="rl_%s_upper" % rid)

                neg_constraint = model.solver.interface.Constraint(
                    (1 - x) * reaction.lower_bound - reaction.flux_expression,
                    ub=0, name="rl_%s_lower" % rid)

                constraints.extend([pos_constraint, neg_constraint])

                x_weights.append(abs(weight))

        for variable in x_variables:
            model.solver.add(variable)

        for variables in y_variables:
            model.solver.add(variables[0])
            model.solver.add(variables[1])

        for constraint in constraints:
            model.solver.add(constraint)

        rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
        rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
        objective = model.solver.interface.Objective(Add(*rh_objective) + Add(*rl_objective), direction="max")

        with model:
            model.objective = objective
            solution = model.optimize()
            return solution

    finally:
        #model.solver.remove([var for var in binary_variables if var in model.solver.variables])
        #model.solver.remove([const for const in binary_constraints if const in model.solver.constraints])

        model.solver.remove([var for var in x_variables if var in model.solver.variables])
        model.solver.remove([var for pair in y_variables for var in pair if var in model.solver.variables])
        model.solver.remove([const for const in constraints if const in model.solver.constraints])

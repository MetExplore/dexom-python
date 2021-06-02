
import six
import time
import numpy as np
from sympy import sympify, Add
from src.imat import imat
from src.result_functions import get_binary_sol
from src.enum_functions.enumeration import EnumSolution
from src.enum_functions.icut import create_icut_constraint


def create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol, name="maxdist_optimality", full=False):

    y_variables = []
    y_weights = []
    x_variables = []
    x_weights = []

    if full:
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
    else:
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:
                y_neg = model.solver.variables["rh_" + rid + "_neg"]
                y_pos = model.solver.variables["rh_" + rid + "_pos"]
                y_variables.append([y_neg, y_pos])
                y_weights.append(weight)
            elif weight < 0:
                x_variables.append(model.solver.variables["rl_" + rid])
                x_weights.append(abs(weight))

    lower_opt = prev_sol.objective_value - obj_tol
    rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
    rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
    opt_const = model.solver.interface.Constraint(Add(*rh_objective) + Add(*rl_objective), lb=lower_opt, name=name)
    return opt_const


def create_maxdist_objective(model, reaction_weights, prev_sol, prev_sol_bin, only_ones=False, full=False):
    expr = sympify("0")
    if full:
        for rxn in model.reactions:
            rid = rxn.id
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            x = model.solver.variables["x_" + rid]
            if prev_sol_bin[rid_loc] == 1:
                expr += x
            elif not only_ones:
                expr += 1 - x
    else:
        for rid, weight in six.iteritems(reaction_weights):
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            if weight > 0:
                y_neg = model.solver.variables["rh_" + rid + "_neg"]
                y_pos = model.solver.variables["rh_" + rid + "_pos"]
                if prev_sol_bin[rid_loc] == 1:
                    expr += y_neg + y_pos
                elif not only_ones:
                    expr += 1 - (y_neg + y_pos)
            elif weight < 0:
                x_rl = model.solver.variables["rl_" + rid]
                if prev_sol_bin[rid_loc] == 1:
                    expr += 1 - x_rl
                elif not only_ones:
                    expr += x_rl
    objective = model.solver.interface.Objective(expr, direction="min")
    return objective


def maxdist(model, reaction_weights, prev_sol, threshold=1e-4, obj_tol=1e-3, maxiter=10, full=False, only_ones=False):
    """
    Parameters
    maximal distance enumeration
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: Solution instance
        a previous imat solution
    threshold: float
        detection threshold of activated reactions
    obj_tol: float
        variance allowed in the objective_values of the solutions
    maxiter: foat
        maximum number of solutions to check for
    only_ones: bool
        determines if the hamming distance is only calculated with ones, or with ones & zeros
    Returns
    -------

    """
    icut_constraints = []
    all_solutions = [prev_sol]
    prev_sol_bin = get_binary_sol(prev_sol, threshold)
    all_binary = [prev_sol_bin]

    # adding the optimality constraint: the new objective value must be equal to the previous objective value
    opt_const = create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol,
                                          name="maxdist_optimality", full=full)
    model.solver.add(opt_const)
    for i in range(maxiter):
        t0 = time.perf_counter()
        # adding the icut constraint to prevent the algorithm from finding the same solutions
        const = create_icut_constraint(model, reaction_weights, threshold, prev_sol, prev_sol_bin,
                                       name="icut_"+str(i), full=full)
        model.solver.add(const)
        icut_constraints.append(const)
        # defining the objective: minimize the number of overlapping ones and zeros
        objective = create_maxdist_objective(model, reaction_weights, prev_sol, prev_sol_bin, only_ones, full)
        model.objective = objective
        try:
            with model:
                prev_sol = model.optimize()
            prev_sol_bin = get_binary_sol(prev_sol, threshold)
            all_solutions.append(prev_sol)
            all_binary.append(prev_sol_bin)
        except:
            print("An error occured in iter %i of maxdist, check if all feasible solutions have been found" % (i+1))
            break
        t1 = time.perf_counter()
        print("time for iteration "+str(i+1)+": ", t1-t0)

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)
    return solution

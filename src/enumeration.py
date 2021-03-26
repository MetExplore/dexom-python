
from cobra import Model
import numpy as np
from sympy import sympify, Add
import six
import time

from iMAT import imat


class EnumSolution(object):
    def __init__(self,
                 all_solutions, unique_solutions, all_binary, unique_binary, all_reactions=None, unique_reactions=None):
        self.all_solutions = all_solutions
        self.unique_solutions = unique_solutions
        self.all_binary = all_binary
        self.unique_binary = unique_binary
        self.all_reactions = all_reactions
        self.unique_reactions = unique_reactions


def rxn_enum(model, reaction_weights=None, epsilon=1., threshold=1e-1, tlim=None, tol=1e-7, obj_tol=1e-5):
    """
    Reaction enumeration method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    epsilon: float
        activation threshold in imat
    threshold: float
        detection threshold of activated reactions
    tlim: int
        time limit for imat
    tol: float
        tolerance for imat
    obj_tol: float
        variance allowed in the objective_values of the solutions

    Returns
    -------
    solution: EnumSolution object

    """
    assert isinstance(model, Model)

    initial_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold, timelimit=tlim, tolerance=tol)
    initial_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in initial_solution.fluxes]
    optimal_objective_value = initial_solution.objective_value - obj_tol

    all_solutions = [initial_solution]
    all_solutions_binary = [initial_solution_binary]
    unique_solutions = [initial_solution]
    unique_solutions_binary = [initial_solution_binary]
    all_reactions = []  # for each solution, save which reaction was activated/inactived by the algorithm
    unique_reactions = []

    for idx, reaction in enumerate(model.reactions):
        with model as model_temp:
            rxn = model_temp.reactions.get_by_id(reaction.id)
            # for active fluxes, check inactivation
            if initial_solution_binary[idx] == 1:
                rxn.bounds = (0., 0.)
            # for inactive fluxes, check activation
            else:
                upper_bound_temp = rxn.upper_bound
                # for inactive reversible fluxes, check activation in backwards direction
                if rxn.lower_bound < 0.:
                    try:
                        rxn.upper_bound = -threshold
                        temp_sol = imat(model_temp, reaction_weights, epsilon=epsilon, threshold=threshold, timelimit=tlim, tolerance=tol)
                        temp_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in temp_sol.fluxes]

                        if temp_sol.objective_value >= optimal_objective_value:
                            all_solutions.append(temp_sol)
                            all_solutions_binary.append(temp_sol_bin)
                            all_reactions.append(reaction.id+"_backwards")
                            if temp_sol_bin not in unique_solutions_binary:
                                unique_solutions.append(temp_sol)
                                unique_solutions_binary.append(temp_sol_bin)
                                unique_reactions.append(reaction.id+"_backwards")
                    except:
                        print("An error occurred with reaction %s_backwards. "
                              "Check feasibility of the model when this reaction is irreversible." % reaction.id)
                    finally:
                        rxn.upper_bound = upper_bound_temp
                # for all inactive fluxes, check activation in forwards direction
                rxn.lower_bound = threshold
            # for all fluxes: compute solution with new bounds
            try:
                temp_sol = imat(model_temp, reaction_weights, epsilon=epsilon, threshold=threshold, timelimit=tlim, tolerance=tol)
                temp_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in temp_sol.fluxes]
                if temp_sol.objective_value >= optimal_objective_value:
                    all_solutions.append(temp_sol)
                    all_solutions_binary.append(temp_sol_bin)
                    all_reactions.append(reaction.id)
                    if temp_sol_bin not in unique_solutions_binary:
                        unique_solutions.append(temp_sol)
                        unique_solutions_binary.append(temp_sol_bin)
                        unique_reactions.append(reaction.id)
            except:
                print("An error occurred with reaction %s. "
                      "Check feasibility of the model when this reaction is irreversible" % reaction.id)

    solution = EnumSolution(all_solutions, unique_solutions, all_solutions_binary, unique_solutions_binary,
                            all_reactions, unique_reactions)
    return solution


def create_icut_constraint(model, reaction_weights, threshold, prev_sol, prev_sol_binary, full):

    expr = sympify("1")
    if full:
        newbound = sum(prev_sol_binary)
        cvector = [1 if x else -1 for x in prev_sol_binary]
        for idx, rxn in enumerate(model.reactions):
            expr += cvector[idx] * model.solver.variables["x_" + rxn.id]
    else:
        newbound = 0
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0.:
                if prev_sol.fluxes[rid] >= threshold:
                    expr += model.solver.variables["rh_" + rid + "_pos"] - model.solver.variables["rh_" + rid + "_neg"]
                    newbound += 1
                elif prev_sol.fluxes[rid] <= -threshold:
                    expr += model.solver.variables["rh_" + rid + "_neg"] - model.solver.variables["rh_" + rid + "_pos"]
                    newbound += 1
                else:
                    expr += - model.solver.variables["rh_" + rid + "_pos"] - model.solver.variables[
                        "rh_" + rid + "_neg"]
            elif weight < 0.:
                if np.abs(prev_sol.fluxes[rid]) < threshold:
                    expr += model.solver.variables["rl_" + rid]
                    newbound += 1
                else:
                    expr += - model.solver.variables["rl_" + rid]
    return expr, newbound


def icut(model, reaction_weights=None, epsilon=1., threshold=1e-1, tlim=None, tol=1e-7, obj_tol=1e-5, maxiter=10, full=False):
    """
    integer-cut method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    epsilon: float
        activation threshold in imat
    threshold: float
        detection threshold of activated reactions
    tlim: int
        time limit for imat
    tol: float
        tolerance for imat
    obj_tol: float
        variance allowed in the objective_values of the solutions
    maxiter: foat
        maximum number of solutions to check for
    full: bool
        if True, carries out integer-cut on all reactions; if False, only on reactions with non-zero weights

    Returns
    -------
    solution: EnumSolution object
        In the case of integer-cut, all_solutions and unique_solutions are identical
    """

    assert isinstance(model, Model)
    
    new_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold, timelimit=tlim, tolerance=tol, full=full)
    new_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in new_solution.fluxes]
    optimal_objective_value = new_solution.objective_value - obj_tol

    all_solutions = [new_solution]
    all_solutions_binary = [new_solution_binary]
    icut_constraints = []

    for i in range(maxiter):
        t0 = time.perf_counter()

        expr, newbound = create_icut_constraint(model, reaction_weights, threshold, new_solution, new_solution_binary, full)
        if expr.evalf() == 1 and not full:
            print("No reactions were found in reaction_weights when attempting to create an icut constraint")
            break

        newconst = model.solver.interface.Constraint(expr, ub=newbound, name="icut_"+str(i))
        model.solver.add(newconst)
        icut_constraints.append(newconst)

        try:
            new_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold, timelimit=tlim, tolerance=tol, full=full)
        except:
            print("An error occured in iteration %i of icut, check if all feasible solutions have been found" % (i+1))
            break
        t1 = time.perf_counter()
        print("time for iteration "+str(i+1)+": ", t1-t0)

        if new_solution.objective_value >= optimal_objective_value:
            all_solutions.append(new_solution)
            new_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in new_solution.fluxes]
            all_solutions_binary.append(new_solution_binary)
        else:
            break

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    solution = EnumSolution(all_solutions, all_solutions, all_solutions_binary, all_solutions_binary)
    if full:
        print("full icut iterations: ", i+1)
    else:
        print("partial icut iterations: ", i+1)
    return solution


def maxdist(model, reaction_weights, epsilon=1., threshold=1e-4, tlim=None, tol=1e-7, obj_tol=1e-3, maxiter=10):

    full = False
    prev_sol = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold, timelimit=tlim, tolerance=tol, full=full)

    icut_constraints = []
    all_solutions = [prev_sol]
    prev_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in prev_sol.fluxes]
    all_binary = [prev_sol_bin]

    # adding the optimality constraint: the new objective value must be equal to the previous objective value
    # possibility: count the number of ones in the vector, instead of using the reaction weights

    y_variables = []
    y_weights = []
    x_variables = []
    x_weights = []

    for rid, weight in six.iteritems(reaction_weights):
        if weight > 0:  # the rh_rid variables represent the highly expressed reactions
            y_neg = model.solver.variables["rh_" + rid + "_neg"]
            y_pos = model.solver.variables["rh_" + rid + "_pos"]
            y_variables.append([y_neg, y_pos])
            y_weights.append(weight)

        elif weight < 0:  # the rl_rid variables represent the lowly expressed reactions
            x_variables.append(model.solver.variables["rl_" + rid])
            x_weights.append(abs(weight))

    lower_opt = prev_sol.objective_value - obj_tol
    upper_opt = prev_sol.objective_value + obj_tol

    rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
    rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
    opt_const = model.solver.interface.Constraint(Add(*rh_objective) + Add(*rl_objective),
                                              lb=lower_opt, ub=upper_opt, name="optimality")
    model.solver.add(opt_const)

    for i in range(maxiter):
        t0 = time.perf_counter()
        #adding the icut constraint to prevent the algorithm from finding the same solutions

        expr, newbound = create_icut_constraint(model, reaction_weights, threshold, prev_sol, prev_sol_bin, full)
        newconst = model.solver.interface.Constraint(expr, ub=newbound, name="icut_" + str(i))
        model.solver.add(newconst)
        icut_constraints.append(newconst)

        # defining the objective: minimize the number of overlapping ones and zeros
        expr = sympify("0")
        for rid, weight in six.iteritems(reaction_weights):
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            if weight > 0:
                y_neg = model.solver.variables["rh_" + rid + "_neg"]
                y_pos = model.solver.variables["rh_" + rid + "_pos"]
                # expr += (y_neg + y_pos) * prev_sol_bin[rid_loc]
                if prev_sol_bin[rid_loc] == 1:
                    expr += y_neg + y_pos
                else:
                    expr += 1 - (y_neg + y_pos)
            elif weight < 0:
                x = model.solver.variables["rl_" + rid]
                # expr += x * prev_sol_bin[rid_loc]
                if prev_sol_bin[rid_loc] == 1:
                    expr += x
                else:
                    expr += 1 - x
        objective = model.solver.interface.Objective(expr, direction="min")
        model.objective = objective
        try:
            with model:
                prev_sol = model.optimize()
            prev_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in prev_sol.fluxes]
            all_solutions.append(prev_sol)
            all_binary.append(prev_sol_bin)
        except:
            print("An error occured in iteration %i of maxdist, check if all feasible solutions have been found" % (i+1))
            break
        t1 = time.perf_counter()
        print("time for iteration "+str(i+1)+": ", t1-t0)

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)

    solution = EnumSolution(all_solutions, all_solutions, all_binary, all_binary)

    return solution


from cobra import Model
from iMAT import imat
import numpy as np
from sympy import sympify
import six


class EnumSolution(object):
    def __init__(self,
                 all_solutions, unique_solutions, all_binary, unique_binary, all_reactions=None, unique_reactions=None):
        self.all_solutions = all_solutions
        self.unique_solutions = unique_solutions
        self.all_binary = all_binary
        self.unique_binary = unique_binary
        self.all_reactions = all_reactions
        self.unique_reactions = unique_reactions


def rxn_enum(model, reaction_weights=None, epsilon=0.1, threshold=1e-3):
    """
    Reaction enumeration method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict with keys = reactions and values = weights
    epsilon: float, activation threshold in imat
    threshold: float, detection threshold of activated reactions

    Returns
    -------
    solution: EnumSolution object

    """
    assert isinstance(model, Model)

    initial_solution = imat(model, reaction_weights, epsilon, threshold)
    initial_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in initial_solution.fluxes]
    optimal_objective_value = initial_solution.objective_value

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
                        rxn.upper_bound = -epsilon
                        constraint_name = "x_"+rxn.id+"_zero"
                        #model_temp.solver.constraints[constraint_name].ub = - rxn.lower_bound / threshold
                        #model_temp.solver.constraints[constraint_name].lb = 0.

                        temp_sol = imat(model_temp, reaction_weights, epsilon, threshold)
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
                        constraint_name = "x_"+rxn.id+"_zero"
                        #model_temp.solver.constraints[constraint_name].ub = 0.
                        #model_temp.solver.constraints[constraint_name].lb = - rxn.upper_bound / threshold
                # for all inactive fluxes, check activation in forwards direction
                rxn.lower_bound = epsilon
            # for all fluxes: compute solution with new bounds
            try:
                temp_sol = imat(model_temp, reaction_weights, epsilon, threshold)
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


def full_icut(model, reaction_weights=None, epsilon=0.1, threshold=1e-3, maxiter=10):

    assert isinstance(model, Model)

    new_solution = imat(model, reaction_weights, epsilon, threshold)
    new_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in new_solution.fluxes]
    optimal_objective_value = new_solution.objective_value

    all_solutions = [new_solution]
    all_solutions_binary = [new_solution_binary]

    for i in range(maxiter):
        newbound = sum(new_solution_binary)
        cvector = [1 if x else -1 for x in new_solution_binary]
        expr = sympify("1")
        for idx, rxn in enumerate(model.reactions):
            expr += cvector[idx] * model.solver.variables["x_"+rxn.id]
        newconst = model.solver.interface.Constraint(expr, ub=newbound, name="icut_"+str(i))
        model.solver.add(newconst)

        new_solution = imat(model, reaction_weights, epsilon, threshold)

        if new_solution.objective_value >= optimal_objective_value:
            all_solutions.append(new_solution)
            new_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in new_solution.fluxes]
            all_solutions_binary.append(new_solution_binary)
        else:
            break

    solution = EnumSolution(all_solutions, all_solutions, all_solutions_binary, all_solutions_binary)
    print("number of iterations: ", i+1)
    return solution


def partial_icut(model, reaction_weights=None, epsilon=0.1, threshold=1e-3, maxiter=10):

    assert isinstance(model, Model)

    new_solution = imat(model, reaction_weights, epsilon, threshold)
    new_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in new_solution.fluxes]
    optimal_objective_value = new_solution.objective_value

    all_solutions = [new_solution]
    all_solutions_binary = [new_solution_binary]

    for i in range(maxiter):
        expr = sympify("1")
        newbound = 0
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0.:
                if new_solution.fluxes[rid] >= epsilon:
                    expr += model.solver.variables ["rh_"+rid+"_pos"] - model.solver.variables ["rh_"+rid+"_neg"]
                    newbound += 1
                elif new_solution.fluxes[rid] <= -epsilon:
                    expr += model.solver.variables["rh_" + rid + "_neg"] - model.solver.variables["rh_" + rid + "_pos"]
                    newbound += 1
                else:
                    expr += - model.solver.variables ["rh_"+rid+"_pos"] - model.solver.variables ["rh_"+rid+"_neg"]
            elif weight < 0.:
                if np.abs(new_solution.fluxes[rid]) < threshold:
                    expr += model.solver.variables ["rl_"+rid]
                    newbound += 1
                else:
                    expr += - model.solver.variables ["rl_"+rid]
        newconst = model.solver.interface.Constraint(expr, ub=newbound, name="icut_"+str(i))
        model.solver.add(newconst)

        new_solution = imat(model, reaction_weights, epsilon, threshold)

        if new_solution.objective_value >= optimal_objective_value:
            all_solutions.append(new_solution)
            new_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in new_solution.fluxes]
            all_solutions_binary.append(new_solution_binary)
        else:
            break

    solution = EnumSolution(all_solutions, all_solutions, all_solutions_binary, all_solutions_binary)
    print("number of iterations: ", i+1)
    return solution

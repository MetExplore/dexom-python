
from cobra import Model
from iMAT import imat
import numpy as np


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

    initial_solution = imat(model, reaction_weights, epsilon)
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
            if initial_solution_binary[idx] == 1:
                rxn.bounds = (0., 0.)
            else:
                upper_bound_temp = rxn.upper_bound
                if rxn.lower_bound < 0.:
                    # for reversible fluxes
                    try:
                        rxn.upper_bound = -epsilon
                        temp_sol = imat(model_temp, reaction_weights, epsilon)
                        temp_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in temp_sol.fluxes]
                        if temp_sol.objective_value == optimal_objective_value:
                            all_solutions.append(temp_sol)
                            all_solutions_binary.append(temp_sol_bin)
                            all_reactions.append(reaction.id+"_backwards")
                            if temp_sol_bin not in unique_solutions_binary:
                                unique_solutions.append(temp_sol)
                                unique_solutions_binary.append(temp_sol_bin)
                                unique_reactions.append(reaction.id+"_backwards")
                    except:
                        print("An error occurred with reaction %s_backwards. Check feasibility of the model" % reaction.id)
                rxn.upper_bound = upper_bound_temp
                rxn.lower_bound = epsilon
            try:
                temp_sol = imat(model_temp, reaction_weights, epsilon)
                temp_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in temp_sol.fluxes]
                if temp_sol.objective_value == optimal_objective_value:
                    all_solutions.append(temp_sol)
                    all_solutions_binary.append(temp_sol_bin)
                    all_reactions.append(reaction.id)
                    if temp_sol_bin not in unique_solutions_binary:
                        unique_solutions.append(temp_sol)
                        unique_solutions_binary.append(temp_sol_bin)
                        unique_reactions.append(reaction.id)
            except:
                print("An error occurred with reaction %s. Check feasibility of the model" % reaction.id)

    solution = EnumSolution(all_solutions, unique_solutions, all_solutions_binary, unique_solutions_binary,
                            all_reactions, unique_reactions)
    return solution


def icut(model, reaction_weights=None, epsilon=0.1, threshold=1e-3, maxiter = 100, maxsolutions = 100, maxunique = 5):

    assert isinstance(model, Model)

    previous_solution = imat(model, reaction_weights, epsilon)
    previous_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in initial_solution.fluxes]
    optimal_objective_value = previous_solution.objective_value

    all_solutions = [previous_solution]
    all_solutions_binary = [previous_solution_binary]
    unique_solutions = [previous_solution]
    unique_solutions_binary = [previous_solution_binary]

    iter = 0
    numsol = 1
    numuni = 1
    while iter < maxiter and numsol < maxsolutions and numuni < maxunique:
        iter += 1

    return 0

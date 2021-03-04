
from cobra import Model
from iMAT import imat
import numpy as np


class EnumSolution(object):
    def __init__(self, all_solutions, unique_solutions, all_binary, unique_binary, all_reactions=None, unique_reactions=None):
        self.all_solutions = all_solutions
        self.unique_solutions = unique_solutions
        self.all_binary = all_binary
        self.unique_binary = unique_binary
        self.all_reactions = all_reactions
        self.unique_reactions = unique_reactions


def rxn_enum(model, reaction_weights=None, epsilon=0.1, threshold=1e-3):

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
                if rxn.lower_bound < 0.:
                    # for reversible fluxes
                    pass
                rxn.lower_bound = threshold
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
    # solution = {'all': all_solutions, 'all binary': all_solutions_binary, 'unique': unique_solutions, 'unique binary':
    #             unique_solutions_binary, 'all reactions': all_reactions, 'unique reactions': unique_reactions}
    solution = EnumSolution(all_solutions, unique_solutions, all_solutions_binary, unique_solutions_binary,
                               all_reactions, unique_reactions)
    return solution

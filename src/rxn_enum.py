
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from model_functions import load_reaction_weights
from result_functions import read_solution, get_binary_sol
from imat import imat
from enumeration import RxnEnumSolution


def partial_rxn_enum(model, rxn_list, init_sol, reaction_weights=None, epsilon=1., threshold=1e-1, tlim=None,
                     feas=1e-6, mipgap=1e-3, obj_tol=1e-5):
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
    init_sol_bin = get_binary_sol(init_sol, threshold)
    optimal_objective_value = init_sol.objective_value - obj_tol

    all_solutions = [initial_solution]
    all_solutions_binary = [init_sol_bin]
    unique_solutions = [initial_solution]
    unique_solutions_binary = [init_sol_bin]
    all_reactions = []  # for each solution, save which reaction was activated/inactived by the algorithm
    unique_reactions = []

    for idx, rid in enumerate(rxn_list):
        with model as model_temp:
            if rid in model.reactions:
                rxn = model_temp.reactions.get_by_id(rid)
                # for active fluxes, check inactivation
                if init_sol_bin[idx] == 1:
                    rxn.bounds = (0., 0.)
                # for inactive fluxes, check activation
                else:
                    upper_bound_temp = rxn.upper_bound
                    # for inactive reversible fluxes, check activation in backwards direction
                    if rxn.lower_bound < 0.:
                        try:
                            rxn.upper_bound = -threshold
                            temp_sol = imat(model_temp, reaction_weights, epsilon=epsilon,
                                            threshold=threshold, timelimit=tlim, feasibility=feas, mipgaptol=mipgap)
                            temp_sol_bin = get_binary_sol(temp_sol, threshold)
                            if temp_sol.objective_value >= optimal_objective_value:
                                all_solutions.append(temp_sol)
                                all_solutions_binary.append(temp_sol_bin)
                                if temp_sol_bin not in unique_solutions_binary:
                                    unique_solutions.append(temp_sol)
                                    unique_solutions_binary.append(temp_sol_bin)
                                    unique_reactions.append(rid+"_backwards")
                        except:
                            print("An error occurred with reaction %s_backwards. "
                                  "Check feasibility of the model when this reaction is irreversible." % rid)
                        finally:
                            rxn.upper_bound = upper_bound_temp
                    # for all inactive fluxes, check activation in forwards direction
                    rxn.lower_bound = threshold
                # for all fluxes: compute solution with new bounds
                try:
                    temp_sol = imat(model_temp, reaction_weights, epsilon=epsilon,
                                    threshold=threshold, timelimit=tlim, feasibility=feas, mipgaptol=mipgap)
                    temp_sol_bin = [1 if np.abs(flux) >= threshold else 0 for flux in temp_sol.fluxes]
                    if temp_sol.objective_value >= optimal_objective_value:
                        all_solutions.append(temp_sol)
                        all_solutions_binary.append(temp_sol_bin)
                        all_reactions.append(rid)
                        if temp_sol_bin not in unique_solutions_binary:
                            unique_solutions.append(temp_sol)
                            unique_solutions_binary.append(temp_sol_bin)
                            unique_reactions.append(rid)
                except:
                    print("An error occurred with reaction %s. "
                          "Check feasibility of the model when this reaction is blocked/irreversible" % rid)

    solution = RxnEnumSolution(all_solutions, unique_solutions, all_solutions_binary, unique_solutions_binary,
                               all_reactions, unique_reactions)

    return solution


if __name__ == "__main__":
    description = "Performs the reaction enumeration algorithm on a specified list of reactions"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in sbml, matlab, or json format")
    parser.add_argument("-l", "--reaction_list", default=None, help="csv list of reactions to enumerate")
    parser.add_argument("--range", default="0_",
                        help="range of reactions to use from the list, in the format 'int_int'")
    parser.add_argument("-r", "--reaction_weights", default=None,
                        help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-i", "--initial_solution", default=None, help="initial imat solution in .txt format")
    parser.add_argument("--epsilon", type=float, default=1e-2,
                        help="Activation threshold for highly expressed reactions")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Activation threshold for all reactions")
    parser.add_argument("-t", "--timelimit", type=int, default=None, help="Solver time limit")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver feasibility tolerance")
    parser.add_argument("--mipgap", type=float, default=1e-3, help="Solver MIP gap tolerance")
    parser.add_argument("--obj_tol", type=float, default=1e-3, help="objective function tolerance")
    parser.add_argument("-o", "--output", default="rxn_enum", help="Name of the output file")
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

    try:
        model.solver = 'cplex'
    except:
        print("cplex is not available or not properly installed")

    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    rxn_list = []
    if args.reaction_list:
        df = pd.read_csv(args.reaction_list, header=None)
        reactions = [x for x in df.unstack().values]
        rxn_range = args.range.split("_")
        if rxn_range[0] == '':
            start = 0
        else:
            start = int(rxn_range[0])
        if rxn_range[1] == '':
            rxn_list = reactions[start:]
        elif int(rxn_range[1]) > len(rxn_list):
            rxn_list = reactions[start:]
        else:
            rxn_list = reactions[start:int(rxn_range[1])]

    if args.initial_solution:
        initial_solution, initial_binary = read_solution(args.initial_solution)
    else:
        initial_solution = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold,
                                timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)

    solution = partial_rxn_enum(model, rxn_list, initial_solution, reaction_weights, args.epsilon,
                                        args.threshold, args.timelimit, args.tol, args.mipgap, args.obj_tol)

    pd.DataFrame(solution.unique_binary).to_csv(args.output+"_solutions.csv")
    result.to_csv(args.output + "_results.csv")

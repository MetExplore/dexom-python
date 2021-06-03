
import argparse
import numpy as np
from pathlib import Path
import time
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from src.model_functions import load_reaction_weights, clean_model
from src.result_functions import get_binary_sol
from src.imat import imat


def permutation_test(model, reaction_weights=None, nperm=10, epsilon=1, threshold=1e-2, timelimit=None, tolerance=1e-6,
                     mipgaptol=1e-3):
    rng = np.random.default_rng()
    if not reaction_weights:
        reaction_weights = {}
    permutation_weights = []
    permutation_solutions = []
    for i in range(nperm):
        t1 = time.perf_counter()
        weights = np.array(list(reaction_weights.values()))
        weights = rng.permutation(weights)
        reaction_weights = dict(zip(reaction_weights.keys(), list(weights)))
        try:
            solution = imat(model, reaction_weights, epsilon, threshold, timelimit, tolerance, mipgaptol)
            solution_binary = get_binary_sol(solution, threshold=threshold)
            permutation_solutions.append(solution_binary)
            permutation_weights.append(list(weights))
        except:
            print("iteration %i failed" % i)
        t2 = time.perf_counter()
        print("iteration %i time: " % i, t2-t1)
        clean_model(model)

    return permutation_solutions, permutation_weights


if __name__ == "__main__":
    description = "Performs weight permutation tests on imat"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in sbml, json, or matlab format")
    parser.add_argument("-r", "--reaction_weights", default={},
                        help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-n", "--num_permutations", type=int, default=10, help="number of permutations to perform")
    parser.add_argument("--epsilon", type=float, default=1e-2, help="Activation threshold for highly expressed reactions")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Activation threshold for all reactions")
    parser.add_argument("-t", "--timelimit", type=int, default=None, help="Solver time limit")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver feasibility tolerance")
    parser.add_argument("--mipgap", type=float, default=1e-3, help="Solver MIP gap tolerance")
    parser.add_argument("-o", "--output", default="imat_solution.txt", help="Name of the output file")

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

    perm_sol, perm_weights = permutation_test(model, reaction_weights, args.num_permutations,
                                              args.epsilon, args.threshold, args.timelimit, args.tol, args.mipgap)

    name = args.output.split(".")
    outname = name.pop(0)
    outsuffix = ".".join(name)

    out_solution = outname+"_solutions."+outsuffix
    out_weights = outname+"_weights."+outsuffix

    reaction_list = [rxn.id for rxn in model.reactions]

    with open(out_solution, "w+") as file:
        file.write(",".join(map(str, reaction_list))+"\n")
        for sol in perm_sol:
            file.write(",".join(map(str, sol))+"\n")
    with open(out_weights, "w+") as file:
        file.write(",".join(map(str, reaction_list)) + "\n")
        for wei in perm_weights:
            file.write(",".join(map(str, wei))+"\n")

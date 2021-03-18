
import argparse
from cobra.io import load_json_model, read_sbml_model
from models import load_reaction_weights, read_solution, write_solution
from iMAT import imat
import numpy as np
from pathlib import Path
import time


def permutation_test(model, reaction_weights={}, nperm = 10, epsilon=1., threshold=1e-1):
    rng = np.random.default_rng()
    permutation_weights = []
    permutation_solutions = []
    for i in range(nperm):
        t1 = time.perf_counter()
        weights = np.array(list(reaction_weights.values()))
        weights[weights != 0] = rng.permutation(weights[weights != 0])
        reaction_weights = dict(zip(reaction_weights.keys(), list(weights)))
        solution = imat(model, reaction_weights, epsilon, threshold)
        solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]
        permutation_solutions.append(solution_binary)
        permutation_weights.append(list(weights))
        t2 = time.perf_counter()
        print("iteration %i time: " % i, t2-t1)

    return permutation_solutions, permutation_weights



if __name__=="__main__":
    description = "Performs weight permutation tests on imat"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in sbml or json format")
    parser.add_argument("-r", "--reaction_weights", default={},
                        help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-n", "--num_permutations", type=int, default=10, help="number of permutations to perform")
    parser.add_argument("--epsilon", type=float, default=1., help="Activation threshold for highly expressed reactions")
    parser.add_argument("--threshold", type=float, default=1e-1, help="Activation threshold for all reactions")
    parser.add_argument("-t", "--timelimit", type=int, default=None, help="Solver time limit")
    parser.add_argument("--tol", type=float, default=1e-7, help="Solver tolerance")
    parser.add_argument("-o", "--output", default="imat_solution.txt", help="Name of the output file")

    args = parser.parse_args()

    fileformat = Path(args.model).suffix
    if fileformat == ".sbml" or fileformat == ".xml":
        model = read_sbml_model(args.model)
    elif fileformat == '.json':
        model = load_json_model(args.model)
    else:
        print("Only SBML and JSON formats are supported for the models")
        model = None

    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    perm_sol, perm_weights = permutation_test(model, reaction_weights, args.num_permutations,
                                              args.epsilon, args.threshold)

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

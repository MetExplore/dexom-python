
import argparse
import six
import time
import numpy as np
import pandas as pd
from pathlib import Path
from cobra.io import load_json_model, load_matlab_model, read_sbml_model
from src.imat import imat, create_partial_variables, create_full_variables
from src.result_functions import read_solution, get_binary_sol, write_solution
from src.model_functions import load_reaction_weights
from enumeration import EnumSolution, get_recent_solution_and_iteration
from icut import create_icut_constraint
from maxdist import create_maxdist_constraint, create_maxdist_objective


def diversity_enum(model, reaction_weights, prev_sol, thr=1e-5, eps=1e-2, obj_tol=1e-3, maxiter=10, dist_anneal=0.995,
                   out_path="enum_dexom", icut=True, full=False, save=False):
    """
    diversity-based enumeration
    Parameters
    ----------
    model
    reaction_weights
    prev_sol
    thr
    obj_tol
    maxiter
    dist_anneal
    icut
    only_ones

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
    dist_anneal: float
        parameter which influences the probability of selecting reactions
    icut: bool
        determines whether icut constraints are applied or not
    only_ones: bool
        determines if the hamming distance is only calculated with ones, or with ones & zeros
    Returns
    -------

    """
    times = []
    selected_recs = []
    prev_sol_bin = get_binary_sol(prev_sol, thr)
    all_solutions = [prev_sol]
    all_binary = [prev_sol_bin]
    icut_constraints = []

    for rid in reaction_weights.keys():
        if reaction_weights[rid] == 0:
            pass
        elif "rh_"+rid+"_pos" not in model.solver.variables and "rl_"+rid not in model.solver.variables:
            model = create_partial_variables(model=model, reaction_weights=reaction_weights, epsilon=eps)
            break

    # preserve the optimality of the solution
    opt_const = create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol, "dexom_optimality", full=full)
    model.solver.add(opt_const)
    model.solver.presolve = True
    for idx in range(1, maxiter+1):
        t0 = time.perf_counter()
        if icut:
            # adding the icut constraint to prevent the algorithm from finding duplicate solutions
            const = create_icut_constraint(model, reaction_weights, thr, prev_sol, prev_sol_bin, "icut_"+str(idx), full)
            model.solver.add(const)
            icut_constraints.append(const)

        # randomly selecting reactions which were active in the previous solution
        tempweights = {}
        i = 0
        for rid, weight in six.iteritems(reaction_weights):
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            if np.random.random() > dist_anneal**idx:
                tempweights[rid] = weight
                i += 1
        selected_recs.append(i)

        objective = create_maxdist_objective(model, tempweights, prev_sol, prev_sol_bin, full=full)
        model.objective = objective
        try:
            t2 = time.perf_counter()
            print("time before optimizing in iteration "+str(idx)+":", t2-t0)
            with model:
                prev_sol = model.optimize()
            prev_sol_bin = get_binary_sol(prev_sol, thr)
            all_solutions.append(prev_sol)
            all_binary.append(prev_sol_bin)
            if save:
                write_solution(prev_sol, thr, filename=out_path+"_solution_"+time.strftime("%Y%m%d-%H%M%S")+".csv")
            t1 = time.perf_counter()
            print("time for optimizing in iteration " + str(idx) + ":", t1 - t2)
            times.append(t1 - t0)
        except:
            print("An error occured in iteration %i of dexom, check if all feasible solutions have been found" % idx)
            times.append(0.)
            break

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)

    df = pd.DataFrame({"selected reactions": selected_recs, "time": times})
    df.to_csv(out_path+"_results.csv")

    sol = pd.DataFrame(solution.binary)
    sol.to_csv(out_path+"_solutions.csv")

    return solution


if __name__ == "__main__":
    description = "Performs the reaction enumeration algorithm on a specified list of reactions"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in sbml, matlab, or json format")
    parser.add_argument("-r", "--reaction_weights", default=None,
                        help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-p", "--prev_sol", default=None, help="starting solution or directory of recent solutions")
    parser.add_argument("--epsilon", type=float, default=1e-2,
                        help="Activation threshold for highly expressed reactions")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Activation threshold for all reactions")
    parser.add_argument("-t", "--timelimit", type=int, default=None, help="Solver time limit")
    parser.add_argument("-i", "--maxiter", type=int, default=10, help="Iteration limit")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver feasibility tolerance")
    parser.add_argument("--mipgap", type=float, default=1e-3, help="Solver MIP gap tolerance")
    parser.add_argument("--obj_tol", type=float, default=1e-3, help="objective function tolerance")
    parser.add_argument("-o", "--output", default="div_enum", help="Base name of output files, without format")
    parser.add_argument("-a", "--dist_anneal", type=float, default=0.995, help="annealing distance")
    parser.add_argument("-s", "--startsol_num", type=int, default=100, help="number of starting solutions")
    parser.add_argument("--noicut", action='store_true', help="Use this flag to remove the icut constraint")
    parser.add_argument("--full", action='store_true', help="Use this flag to assign non-zero weights to all reactions")
    parser.add_argument("--save", action='store_true', help="Use this flag to save each individual solution")
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

    a = args.dist_anneal
    if "." in args.prev_sol:
        prev_sol, prev_bin = read_solution(args.prev_sol)
        model = create_partial_variables(model, reaction_weights, epsilon=args.epsilon)
    elif args.prev_sol:
        prev_sol, i = get_recent_solution_and_iteration(args.prev_sol)
        a = a ** i
        model = create_partial_variables(model, reaction_weights, epsilon=args.epsilon)
    else:
        prev_sol = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold,
                        timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)

    icut = False
    if args.noicut:
      icut = True

    save = False
    if args.save:
        save = True

    if args.full:
        for rxn in model.reactions:
            if rxn.id not in reaction_weights:
                reaction_weights[rxn.id] = args.obj_tol*1e-5
            elif reaction_weights[rxn.id] == 0:
                reaction_weights[rxn.id] = args.obj_tol*1e-5

    dexom_sol = diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol, thr=args.threshold,
                               maxiter=args.maxiter, obj_tol=args.obj_tol, dist_anneal=a, icut=icut,
                               out_path=args.output, full=False, save=save)


# if __name__ == "__main__":
#     # write_batch_script(100)
#
#     from cobra.io import load_json_model, read_sbml_model, load_matlab_model
#     from src.model_functions import load_reaction_weights
#
#     model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
#     reaction_weights = load_reaction_weights("min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv")
#
#     # model = load_matlab_model("recon2_2/recon2v2_corrected.mat")
#     # reaction_weights = load_reaction_weights("recon2_2/microarray_reactions_test.csv", "reactions", "scores")
#
#     try:
#         model.solver = 'cplex'
#     except:
#         print("cplex is not available or not properly installed")
#
#     icut = False
#     full = False
#     only_ones = False
#
#     for rxn in model.reactions:
#         if rxn.id not in reaction_weights:
#             reaction_weights[rxn.id] = -1e-8
#         elif reaction_weights[rxn.id] == 0:
#             reaction_weights[rxn.id] = -1e-8
#
#     model.solver.configuration.verbosity = 2
#     imat_solution = imat(model, reaction_weights, feasibility=1e-6, timelimit=6000, full=full)
#
#     print("\nstarting dexom")
#     dexom_sol = diversity_enum(model, reaction_weights, imat_solution, maxiter=10, obj_tol=1e-3, dist_anneal=0.999,
#                                icut=icut, only_ones=only_ones, full=full)
#     print("\n")
#
#     ## dexom result analysis
#
#     solutions = dexom_results("enum_dexom_results.csv", "enum_dexom_solutions.csv", "enum_dexom")


import six
import time
import numpy as np
import pandas as pd
from src.imat import imat
from src.result_functions import get_binary_sol
from enumeration import EnumSolution
from icut import create_icut_constraint
from maxdist import create_maxdist_constraint, create_maxdist_objective


def diversity_enum(model, reaction_weights, prev_sol, thr=1e-4, obj_tol=1e-3, maxiter=10, dist_anneal=0.995,
                   icut=True, only_ones=False, full=False):
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

    # preserve the optimality of the solution
    opt_const = create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol, "dexom_optimality", full=full)
    model.solver.add(opt_const)
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
            if (prev_sol_bin[rid_loc] == 1 or not only_ones) and np.random.random() > dist_anneal**idx:
                tempweights[rid] = weight
                i += 1
        selected_recs.append(i)

        objective = create_maxdist_objective(model, tempweights, prev_sol, prev_sol_bin, only_ones, full)
        model.objective = objective
        try:
            t2 = time.perf_counter()
            print("time before optimizing in iteration "+str(idx)+":", t2-t0)
            with model:
                prev_sol = model.optimize()
            prev_sol_bin = get_binary_sol(prev_sol, thr)
            all_solutions.append(prev_sol)
            all_binary.append(prev_sol_bin)
            t1 = time.perf_counter()
            print("time for optimizing in iteration " + str(idx) + ":", t1 - t2)
            times.append(t1 - t0)
        except:
            print("An error occured in iteration %i of dexom, check if all feasible solutions have been found" % idx)
            break

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)

    df = pd.DataFrame({"selected reactions": selected_recs, "time": times})
    df.to_csv("enum_dexom_results.csv")

    sol = pd.DataFrame(solution.binary)
    sol.to_csv("enum_dexom_solutions.csv")

    return solution


if __name__ == "__main__":
    from cobra.io import load_json_model, read_sbml_model, load_matlab_model
    from src.model_functions import load_reaction_weights
    from enumeration import dexom_results

    model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    reaction_weights = load_reaction_weights("min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv")

    # model = load_matlab_model("recon2_2/recon2v2_corrected.mat")
    # reaction_weights = load_reaction_weights("recon2_2/microarray_reactions_test.csv", "reactions", "scores")

    try:
        model.solver = 'cplex'
    except:
        print("cplex is not available or not properly installed")

    icut = False
    full = False
    only_ones = False

    for rxn in model.reactions:
        if rxn.id not in reaction_weights:
            reaction_weights[rxn.id] = 1e-8
        elif reaction_weights[rxn.id] == 0:
            reaction_weights[rxn.id] = 1e-8

    model.solver.configuration.verbosity = 2
    imat_solution = imat(model, reaction_weights, feasibility=1e-7, timelimit=2000, full=full)

    print("\nstarting dexom")
    dexom_sol = diversity_enum(model, reaction_weights, imat_solution, maxiter=10, obj_tol=1e-3, dist_anneal=0.995,
                               icut=icut, only_ones=only_ones, full=full)
    print("\n")

    ## dexom result analysis

    solutions = dexom_results("enum_dexom_results.csv", "enum_dexom_solutions.csv", "enum_dexom_newicut")

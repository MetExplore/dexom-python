
import pandas as pd
from cobra.io import load_json_model
from dexom_python.model_functions import load_reaction_weights
from dexom_python.result_functions import write_solution
from dexom_python.imat import imat
from dexom_python.enum_functions.rxn_enum import rxn_enum
from dexom_python.enum_functions.diversity_enum import diversity_enum

if __name__ == '__main__':
    # for testing DEXOM on a toy example

    model = load_json_model("toy_models/small4M.json")
    reaction_weights = load_reaction_weights("toy_models/small4M_weights.csv")

    eps = 1e-2  # threshold of activity for highly expressed reactions
    thr = 1e-5  # threshold of activity for all reactions
    obj_tol = 1e-2  # variance allowed for the objective_value
    tlim = 600  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-6  # feasibility tolerance for the solver
    mipgap = 1e-3  # mip gap tolerance for the solver
    maxiter = 10  # maximum number of iterations
    dist_anneal = 0.9  # diversity-enumeration parameter

    imat_solution = imat(model=model, reaction_weights=reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim,
                         feasibility=tol, mipgaptol=mipgap)
    write_solution(solution=imat_solution, threshold=thr, filename="toy_models/small4M_imatsol.csv")

    rxn_sol = rxn_enum(model=model, rxn_list=[], prev_sol=imat_solution, reaction_weights=reaction_weights, eps=eps,
             thr=thr, tlim=tlim, feas=tol, mipgap=mipgap, obj_tol=obj_tol)
    pd.DataFrame(rxn_sol.unique_binary).to_csv("toy_models/small4M_rxnenum_solutions.csv")

    div_sol = diversity_enum(model=model, prev_sol=imat_solution, reaction_weights=reaction_weights, eps=eps, thr=thr,
                             obj_tol=obj_tol, maxiter=maxiter, out_path="toy_models/small4M_divenum")

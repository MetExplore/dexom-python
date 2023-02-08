import pandas as pd
from dexom_python.model_functions import read_model, load_reaction_weights, check_model_options, DEFAULT_VALUES
from dexom_python.result_functions import write_solution
from dexom_python.imat_functions import imat
from dexom_python.enum_functions.rxn_enum_functions import rxn_enum
from dexom_python.enum_functions.diversity_enum_functions import diversity_enum


def _main():
    """
    This function is called when you run this script from the commandline.
    It performs iMAT, reaction-enumeration, and diversity-enumeration on a toy model
    """
    model = read_model('toy_models/small4M.json')
    reaction_weights = load_reaction_weights('toy_models/small4M_weights.csv')

    eps = DEFAULT_VALUES['epsilon']  # threshold of activity for highly expressed reactions
    thr = DEFAULT_VALUES['threshold']  # threshold of activity for all reactions
    obj_tol = DEFAULT_VALUES['obj_tol']  # variance allowed for the objective_value
    tlim = DEFAULT_VALUES['timelimit']  # time limit (in seconds) for the imat model.optimisation() call
    tol = DEFAULT_VALUES['tolerance']  # feasibility tolerance for the solver
    mipgap = DEFAULT_VALUES['mipgap']  # mip gap tolerance for the solver
    maxiter = 5  # maximum number of iterations
    dist_anneal = 0.9  # diversity-enumeration parameter

    check_model_options(model, timelimit=tlim, feasibility=tol, mipgaptol=mipgap)

    imat_solution = imat(model=model, reaction_weights=reaction_weights, epsilon=eps, threshold=thr)
    write_solution(model=model, solution=imat_solution, threshold=thr, filename='toy_models/small4M_imatsol.csv')

    rxn_sol = rxn_enum(model=model, rxn_list=[], prev_sol=imat_solution, reaction_weights=reaction_weights, eps=eps,
                       thr=thr, obj_tol=obj_tol)
    uniques = pd.DataFrame(rxn_sol.unique_binary)
    uniques.columns = [r.id for r in model.reactions]
    uniques.to_csv('toy_models/small4M_rxnenum_solutions.csv')

    div_sol, div_res = diversity_enum(model=model, prev_sol=imat_solution, reaction_weights=reaction_weights, eps=eps,
                                      thr=thr, obj_tol=obj_tol, maxiter=maxiter, dist_anneal=dist_anneal)
    div_res.to_csv('toy_models/small4M_divenum_results.csv')
    sol = pd.DataFrame(div_sol.binary, columns=[r.id for r in model.reactions])
    sol.to_csv('toy_models/small4M_divenum_solutions.csv')
    return True


if __name__ == '__main__':
    _main()

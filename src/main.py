if __name__ == '__main__':
    import numpy as np
    import time
    from cobra.io import load_json_model, read_sbml_model

    from models import clean_model, load_reaction_weights, read_solution
    from iMAT import imat
    from enumeration import rxn_enum, icut, maxdist
    from permutation_test import permutation_test
    from example_models import dagNet

    t3 = time.perf_counter()

    model = load_json_model("small4M.json")
    reaction_weights = load_reaction_weights("small4M_weights.csv")

    # model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    # reaction_weights_test = load_reaction_weights("min_iMM1865/min_iMM1865_p53_weights.csv")
    # reaction_weights = {}
    # i = 0
    # for k, v in reaction_weights_test.items():
    #     i+=1
    #     if i<=100:
    #         reaction_weights[k] = v
    #     else:
    #         break

    eps = 1.  # threshold of activity for highly expressed reactions
    thr = 1e-1  # threshold of activity for all reactions
    obj_tol = 1e-5  # variance allowed for the objective_value
    tlim = 100  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-5  # tolerance for the solver
    model.solver = "cplex"

    t0 = time.perf_counter()
    print('import time: ', t0 - t3)

    imat_solution = imat(model, reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim, tolerance=tol, full=False)
    imat_solution_binary = [1 if np.abs(flux) >= thr else 0 for flux in imat_solution.fluxes]
    clean_model(model, reaction_weights)

    t1 = time.perf_counter()
    print('imat time: ', t1-t0)

    icut_sol = icut(model, reaction_weights, epsilon=eps, threshold=thr, tlim=tlim, tol=tol, full=False)
    #binary, weight = permutation_test(model, reaction_weights, nperm=3, epsilon=eps, threshold=thr)
    #clean_model(model, reaction_weights)
    t2 = time.perf_counter()
    print("icut time: ", t2-t1)

    maxdist_sol = maxdist(model, reaction_weights, epsilon=eps, threshold=thr, tlim=tlim, tol=tol)

    t3 = time.perf_counter()
    print("maxdist time: ", t3-t2)




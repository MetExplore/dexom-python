if __name__ == '__main__':
    import numpy as np
    import time
    from cobra.io import load_json_model, read_sbml_model

    from models import clean_model, load_reaction_weights, read_solution
    from iMAT import imat
    from enumeration import rxn_enum, icut
    from permutation_test import permutation_test

    t3 = time.perf_counter()

    model = load_json_model("small4M.json")
    #model = read_sbml_model("min_iMM1865/min_iMM1865.xml")

    reaction_weights = load_reaction_weights("small4M_weights.csv")
    #reaction_weights = load_reaction_weights("min_iMM1865/min_iMM1865_3f_weights.csv")

    eps = 1.  # threshold of activity for highly expressed reactions
    thr = 1e-1  # threshold of activity for all reactions
    obj_tol = 1e-5  # variance allowed for the objective_value
    tlim = 100  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-7  # tolerance for the solver

    t0 = time.perf_counter()

    imat_solution = imat(model, reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim, tolerance=tol)
    imat_solution_binary = [1 if np.abs(flux) >= thr else 0 for flux in imat_solution.fluxes]
    clean_model(model, reaction_weights)

    t1 = time.perf_counter()

    #binary, weight = permutation_test(model, reaction_weights, nperm=3, epsilon=eps, threshold=thr)
    #clean_model(model, reaction_weights)
    t2 = time.perf_counter()


    print('imat time: ', t1-t0)
    print('import time: ', t0 - t3)


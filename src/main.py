if __name__ == '__main__':
    import numpy as np
    import time
    from cobra.io import load_json_model, read_sbml_model, load_matlab_model

    from model_functions import clean_model, load_reaction_weights
    from result_functions import read_solution
    from imat import imat
    from enumeration import get_binary_sol, rxn_enum, icut, maxdist
    from permutation_test import permutation_test
    from example_models import dagNet

    t3 = time.perf_counter()

    # model = load_json_model("small4M.json")
    # reaction_weights = load_reaction_weights("small4M_weights.csv")

    # model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    # reaction_weights_test = load_reaction_weights("min_iMM1865/min_iMM1865_p53_weights.csv")
    # reaction_weights = {}
    # i = 0
    # for k, v in reaction_weights_test.items():
    #     i += 1
    #     if i <= 1000:
    #         reaction_weights[k] = v
    #     else:
    #         break

    # Smodel = read_sbml_model("recon2_2/Recon2.2_Swainton2016.xml")
    Jmodel = read_sbml_model("recon2_2/Recon2.2_reimported2_test.xml")

    Mmodel = load_matlab_model("recon2_2/Recon2.2.mat")

    full_reaction_weights = load_reaction_weights("recon2_2/recon2.2_weights.txt")
    reaction_weights = {}

    for key, value in full_reaction_weights.items():
        if key not in Jmodel.reactions:
            print(key, "not in Jmodel.reactions")

    model = Mmodel

    for key, value in full_reaction_weights.items():
        if key not in model.reactions:
            print(key, " not in model.reactions")
        else:
            # model.reactions.get_by_id(key).bounds = (-1000, 1000)
            if abs(value) > 0.5:
                reaction_weights[key] = value

    eps = 1e-2  # threshold of activity for highly expressed reactions
    thr = 1e-5  # threshold of activity for all reactions
    obj_tol = 1e-5  # variance allowed for the objective_value
    tlim = 600  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-6  # tolerance for the solver
    maxiter = 10
    model.solver = "cplex"

    t0 = time.perf_counter()
    print('import time: ', t0 - t3)

    imat_solution = imat(model, reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim, tolerance=tol, full=False)
    imat_solution_binary = get_binary_sol(imat_solution, thr)
    clean_model(model, reaction_weights)

    t1 = time.perf_counter()
    print('imat time: ', t1-t0)

    # # icut_sol = icut(model, reaction_weights, epsilon=eps, threshold=thr, tlim=tlim, tol=tol, full=False, maxiter=maxiter)
    # #binary, weight = permutation_test(model, reaction_weights, nperm=3, epsilon=eps, threshold=thr)
    # #clean_model(model, reaction_weights)
    # imat2_solution = imat(model, full_reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim, tolerance=tol, full=False)
    # imat2_solution_binary = get_binary_sol(imat_solution, thr)
    # clean_model(model, reaction_weights)
    #
    # t2 = time.perf_counter()
    # print("imat time with all weights: ", t2-t1)

    # maxdist_sol = maxdist(model, reaction_weights, epsilon=eps, threshold=thr, tlim=tlim, tol=tol, maxiter=maxiter)
    #
    # t3 = time.perf_counter()
    # print("maxdist time: ", t3-t2)
    #
    # for idx, solution in enumerate(maxdist_sol.binary):
    #     count = 0
    #     if idx == 0:
    #         pass
    #     else:
    #         test = [0 if x == maxdist_sol.binary[idx-1][i] else 1 for i, x in enumerate(solution)]
    #         newtest = np.count_nonzero(maxdist_sol.binary[idx-1] != maxdist_sol.binary[idx])
    #         print(sum(test))
    #         print(newtest)

if __name__ == '__main__':
    import time
    from cobra.io import read_sbml_model

    from model_functions import load_reaction_weights
    from imat import imat
    from src.enum_functions.enumeration import get_binary_sol

    t3 = time.perf_counter()

    #model = load_json_model("recon2_2/Recon2.2_mat.json")

    model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    reaction_weights = load_reaction_weights("min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv",)

    eps = 1e-2  # threshold of activity for highly expressed reactions
    thr = 1e-5  # threshold of activity for all reactions
    obj_tol = 1e-3  # variance allowed for the objective_value
    tlim = 6000  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-6  # feasibility tolerance for the solver
    mipgap = 1e-3  # mip gap tolerance for the solver
    maxiter = 10
    model.solver = "cplex"
    model.solver.configuration.verbosity = 3
    model.solver.configuration.presolve = True

    t0 = time.perf_counter()
    print('import time: ', t0 - t3)

    imat_solution = imat(model, reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim, feasibility=tol,
                         mipgaptol=mipgap, full=False)
    t1 = time.perf_counter()
    print('total imat time: ', t1-t0)

    imat_solution_binary = get_binary_sol(imat_solution, thr)
    #clean_model(model, reaction_weights)
    print("end of script")

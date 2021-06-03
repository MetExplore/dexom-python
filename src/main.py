if __name__ == '__main__':
    import time
    from cobra.io import read_sbml_model, load_matlab_model, load_json_model, save_json_model

    from src.model_functions import load_reaction_weights
    from src.result_functions import write_solution, read_solution
    from src.imat import imat
    from src.enum_functions.diversity_enum import diversity_enum
    from src.enum_functions.enumeration import dexom_results


    # mat format: 21.1 s on average (100 iterations)
    # json format: 5.9 s on average (100 iterations)
    # sbml format: 6 s on average (100 iterations)

    t3 = time.perf_counter()
    model = load_json_model("recon2_2/recon2v2_corrected.json")
    reaction_weights = load_reaction_weights("recon2_2/microarray_hgnc_pval_0-01_weights.csv")

    # model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    # reaction_weights = load_reaction_weights("min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv",)
    #
    eps = 1e-2  # threshold of activity for highly expressed reactions
    thr = 1e-5  # threshold of activity for all reactions
    obj_tol = 1e-3  # variance allowed for the objective_value
    tlim = 6000  # time limit (in seconds) for the imat model.optimisation() call
    tol = 1e-6  # feasibility tolerance for the solver
    mipgap = 1e-3  # mip gap tolerance for the solver
    maxiter = 10
    model.solver = "cplex"
    # model.solver.configuration.verbosity = 3
    model.solver.configuration.presolve = True
    #
    t0 = time.perf_counter()
    print('import time: ', t0 - t3)
    # #
    # imat_solution = imat(model, reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim, feasibility=tol,
    #                      mipgaptol=mipgap, full=False)
    # t1 = time.perf_counter()
    # print('total imat time: ', t1-t0)
    imat_solution, binary = read_solution("recon2_2/recon_imatsol_pval_0-01.csv")

    div_sol = diversity_enum(model, reaction_weights, prev_sol=imat_solution, maxiter=500, out_path="recon_dexom")

    solutions = dexom_results("recon_dexom_results.csv", "recon_dexom_solutions.csv", "recon_dexom")
    #
    # #clean_model(model, reaction_weights)


if __name__ == '__main__':
    import numpy as np
    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum, icut
    import time
    from cobra.io import load_json_model

    # model, reaction_weights = small4M()
    model = load_json_model("small4M.json")

    reaction_weights = {}
    RH_reactions = ['RFG']
    RL_reactions = ['RAB', 'RDE', 'RCF']
    for rname in RH_reactions:
        reaction_weights[rname] = 1.
    for rname in RL_reactions:
        reaction_weights[rname] = -1.

    epsilon = 1  # threshold of activity for highly expressed reactions in imat, and for bounds in rxn_enum
    threshold = epsilon  # threshold of activity for computing binary solution
    tolerance = 1e-4  # variance allowed for the objective_value

    t0 = time.perf_counter()

    imat_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold)
    imat_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in imat_solution.fluxes]

    t1 = time.perf_counter()

    rxn_solution = rxn_enum(model, reaction_weights, epsilon=epsilon, threshold=threshold, tolerance=tolerance)

    t2 = time.perf_counter()

    part_icut_solution = icut(model, reaction_weights, epsilon=epsilon, threshold=threshold, tolerance=tolerance,
                              maxiter=8, full=False)

    t3 = time.perf_counter()

    full_icut_solution = icut(model, reaction_weights, epsilon=epsilon, threshold=threshold, tolerance=tolerance,
                              maxiter=200, full=True)

    t4 = time.perf_counter()

    print('imat time: ', t1-t0)
    print('rxn-enum time: ', t2 - t1)
    print('partial icut time: ', t3 - t2)
    print('full icut time: ', t4 - t3)

if __name__ == '__main__':
    import numpy as np
    import time
    from cobra.io import load_json_model

    from models import clean_model, load_reaction_weights, read_solution
    from iMAT import imat
    from enumeration import rxn_enum, icut
    from permutation_test import permutation_test

    model = load_json_model("small4M.json")

    reaction_weights = load_reaction_weights("small4M_weights.csv")

    epsilon = 1.  # threshold of activity for highly expressed reactions
    threshold = 1e-1  # threshold of activity for all reactions
    tolerance = 1e-4  # variance allowed for the objective_value

    t0 = time.perf_counter()

    imat_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold)
    imat_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in imat_solution.fluxes]
    clean_model(model, reaction_weights)

    t1 = time.perf_counter()

    bin, wei = permutation_test(model, reaction_weights, nperm=3, epsilon=epsilon, threshold=threshold)
    clean_model(model, reaction_weights)
    t2 = time.perf_counter()

    print(bin)
    print(wei)

    print('imat time: ', t1-t0)
    print('permutation time: ', t2 - t1)

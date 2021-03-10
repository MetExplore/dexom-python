
if __name__ == '__main__':
    import numpy as np
    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum, partial_icut
    import time

    model, reaction_weights = small4M()

    epsilon = 1  # threshold of activity for highly expressed reactions in imat, and for bounds in rxn_enum
    threshold = 1e-1  # threshold of activity for computing binary solution

    t0 = time.perf_counter()

    imat_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold)
    imat_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in imat_solution.fluxes]

    t1 = time.perf_counter()

    rxn_solution = rxn_enum(model, reaction_weights, epsilon=epsilon, threshold=threshold)

    t2 = time.perf_counter()

    part_icut_solution = partial_icut(model, reaction_weights, epsilon=epsilon, threshold=threshold, maxiter=20)

    t3 = time.perf_counter()

    print('imat time: ', t1-t0)
    print('rxn-enum time: ', t2 - t1)
    print('icut time: ', t3 - t2)

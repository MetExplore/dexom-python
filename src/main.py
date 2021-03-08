
if __name__ == '__main__':
    import numpy as np
    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum, icut
    import time

    t0 = time.perf_counter()

    model, reaction_weights = small4M()

    epsilon = 0.1  # threshold of activity for highly expressed reactions in imat, and for bounds in rxn_enum
    threshold = 1e-3  # threshold of activity for computing binary solution

    #solution = imat(model, reaction_weights)
    #binary_solution = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]

    #reaction_weights['R2'] = 3.
    #reaction_weights['R3'] = -2.
    #solution = rxn_enum(model, reaction_weights, epsilon=epsilon, threshold=threshold)

    solution = icut(model, reaction_weights, epsilon=epsilon, threshold=threshold)

    t1 = time.perf_counter()
    print('total time: ', t1-t0)


if __name__ == '__main__':
    import numpy as np

    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum
    import time

    t0 = time.perf_counter()

    model, reaction_weights = small4S()
    threshold = 1e-3  # threshold to consider whether a reaction is active or not

    """
    solution = imat(model, reaction_weights)
    binary_solution = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]
    """

    reaction_weights['R2'] = 3
    reaction_weights['R3'] = -2
    reaction_weights['R9'] = 3
    solution = rxn_enum(model, reaction_weights, threshold=threshold)

    t1 = time.perf_counter()
    print('total time: ', t1-t0)

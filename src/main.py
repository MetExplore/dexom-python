if __name__ == '__main__':
    import numpy as np
    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum, icut
    import time

    model, reaction_weights = small4M()
    threshold = 1.  # threshold of activity for all reactions
    tolerance = 1e-4  # variance allowed for the objective_value

    t0 = time.perf_counter()

    imat_solution = imat(model, reaction_weights, threshold=threshold)
    imat_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in imat_solution.fluxes]

    t1 = time.perf_counter()

    rxn_solution = rxn_enum(model, reaction_weights, threshold=threshold, tolerance=tolerance)

    t2 = time.perf_counter()

    part_icut_solution = icut(model, reaction_weights, threshold=threshold, tolerance=tolerance,
                              maxiter=200, full=False)

    t3 = time.perf_counter()

    full_icut_solution = icut(model, reaction_weights, threshold=threshold, tolerance=tolerance,
                              maxiter=200, full=True)

    t4 = time.perf_counter()

    print('imat time: ', t1-t0)
    print('rxn-enum time: ', t2 - t1)
    print('partial icut time: ', t3 - t2)
    print('full icut time: ', t4 - t3)

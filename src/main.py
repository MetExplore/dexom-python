if __name__ == '__main__':
    import numpy as np
    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum, icut
    import time
    from cobra.io import load_json_model
    import csv

    model = load_json_model("small4M.json")

    with open("small4M_weights.csv", newline="") as file:
        read = csv.DictReader(file)
        for row in read:
            reaction_weights = row
    for k, v in reaction_weights.items():
        reaction_weights[k] = float(v)

    epsilon = 1.  # threshold of activity for highly expressed reactions
    threshold = 1e-1  # threshold of activity for all reactions
    tolerance = 1e-4  # variance allowed for the objective_value

    t0 = time.perf_counter()

    with model:
        imat_solution = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold)
    imat_solution_binary = [1 if np.abs(flux) >= threshold else 0 for flux in imat_solution.fluxes]

    t1 = time.perf_counter()

    with model:
        rxn_solution = rxn_enum(model, reaction_weights, epsilon=epsilon, threshold=threshold, tolerance=tolerance)

    t2 = time.perf_counter()

    with model:
        part_icut_solution = icut(model, reaction_weights, epsilon=epsilon, threshold=threshold, tolerance=tolerance,
                                  maxiter=200, full=False)

    t3 = time.perf_counter()

    with model:
        full_icut_solution = icut(model, reaction_weights, epsilon=epsilon, threshold=threshold, tolerance=tolerance,
                                  maxiter=200, full=True)

    t4 = time.perf_counter()

    print('imat time: ', t1-t0)
    print('rxn-enum time: ', t2 - t1)
    print('partial icut time: ', t3 - t2)
    print('full icut time: ', t4 - t3)

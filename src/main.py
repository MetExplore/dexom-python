
if __name__ == '__main__':
    import numpy as np

    from example_models import small4M, small4S, dagNet
    from iMAT import imat
    from enumeration import rxn_enum

    model, reaction_weights = small4M()
    threshold = 1e-3  # threshold to consider whether a reaction is active or not

    """
    solution = imat(model, reaction_weights)
    binary_solution = [1 if np.abs(flux) >= threshold else 0 for flux in solution.fluxes]
    """

    solution = rxn_enum(model, reaction_weights, threshold=threshold)

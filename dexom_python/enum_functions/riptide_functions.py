import argparse
import time
import cobra.util.array
import optlang
from symengine import Add, sympify
from numpy import abs, max
from statistics import median
from warnings import catch_warnings, filterwarnings, resetwarnings
from cobra.exceptions import OptimizationError
from dexom_python.model_functions import read_model, check_model_options, load_reaction_weights, check_threshold_tolerance, check_constraint_primal_values
from dexom_python.result_functions import write_solution
from dexom_python.default_parameter_values import DEFAULT_VALUES
from dexom_python.imat_functions import _imat_call_model_optimizer


def imat_riptide_like(model, reaction_weights=None, objective_direction='max'):
    """
    Modified version of the integrative Metabolic Analysis Tool with reaction weights

    Parameters
    ----------
    model: cobra.Model
        a cobrapy model
    reaction_weights: dict
        keys are reaction IDs, values are weights
    epsilon: float
        activation threshold for highly expressed reactions
    threshold: float
        activation threshold for all reactions
    full: bool
        if True, create variables for all reactions. if False, only for reactions with non-zero weights

    Returns
    -------
    solution: cobra.Solution
    """
    if reaction_weights is None:
        UserWarning('Without reaction-weights, imat_riptide_like will simply maximize/minimize sums of fluxes')
        reaction_weights = {}
    if objective_direction not in ['min', 'max']:
        UserWarning('Unclear objective_direction, weighted fluxes will be maximized')
        objective_direction = 'max'

    t0 = time.perf_counter()
    med = median(reaction_weights.values())
    rw = {r.id: reaction_weights[r.id] if r.id in reaction_weights.keys() else med for r in model.reactions}

    objective = [(r.forward_variable + r.reverse_variable) * rw[r.id] for r in model.reactions]
    objective = model.solver.interface.Objective(Add(*objective), direction=objective_direction)
    model.objective = objective
    t1 = time.perf_counter()
    print('%.2fs before optimize call' % (t1 - t0))
    solution = _imat_call_model_optimizer(model)
    return solution
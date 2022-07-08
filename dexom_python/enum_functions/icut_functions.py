import six
import time
import numpy as np
from symengine import sympify
from dexom_python.imat_functions import imat
from dexom_python.enum_functions.enumeration import EnumSolution
from dexom_python.enum_functions.enumeration import create_enum_variables


def create_icut_constraint(model, reaction_weights, threshold, prev_sol, name, full=False):
    """
    Creates an icut constraint on the previously found solution.
    This solution is excluded from the solution space.
    """
    tol = model.solver.configuration.tolerances.feasibility
    if full:
        prev_sol_binary = (np.abs(prev_sol.fluxes) >= threshold-tol).values.astype(int)
        expr = sympify('1')
        newbound = sum(prev_sol_binary)
        cvector = [1 if x else -1 for x in prev_sol_binary]
        for idx, rxn in enumerate(model.reactions):
            expr += cvector[idx] * model.solver.variables['x_' + rxn.id]
    else:
        newbound = -1
        var_vals = []
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:
                y = model.solver.variables['rh_' + rid + '_pos']
                x = model.solver.variables['rh_' + rid + '_neg']
                if np.abs(prev_sol.fluxes[rid]) >= threshold-tol:
                    var_vals.append(y + x)
                    newbound += 1
                elif np.abs(prev_sol.fluxes[rid]) < threshold-tol:  # else
                    var_vals.append(-y - x)
            elif weight < 0:
                x = sympify('1') - model.solver.variables['rl_' + rid]  # uses new variable implementation
                if np.abs(prev_sol.fluxes[rid]) < (threshold-tol):
                    var_vals.append(x)
                    newbound += 1
                elif np.abs(prev_sol.fluxes[rid]) >= (threshold-tol):  # else
                    var_vals.append(-x)
        expr = sum(var_vals)
    constraint = model.solver.interface.Constraint(expr, ub=newbound, name=name)
    if expr.evalf() == 1:
        print('No reactions were found in reaction_weights when attempting to create an icut constraint')
        constraint = None
    return constraint


def icut(model, reaction_weights, prev_sol=None, eps=1e-2, thr=1e-5, obj_tol=1e-3, maxiter=10, full=False):
    """
    integer-cut method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: imat Solution object
        an imat solution used as a starting point
    eps: float
        activation threshold in imat
    thr: float
        detection threshold of activated reactions
    obj_tol: float
        variance allowed in the objective_values of the solutions
    maxiter: foat
        maximum number of solutions to check for
    full: bool
        if True, carries out integer-cut on all reactions; if False, only on reactions with non-zero weights
    Returns
    -------
    solution: EnumSolution object
        In the case of integer-cut, all_solutions and unique_solutions are identical
    """
    if prev_sol is None:
        prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=full)
    else:
        model = create_enum_variables(model=model, reaction_weights=reaction_weights, eps=eps, thr=thr, full=full)
    tol = model.solver.configuration.tolerances.feasibility
    prev_sol_binary = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
    optimal_objective_value = prev_sol.objective_value - obj_tol*prev_sol.objective_value

    all_solutions = [prev_sol]
    all_solutions_binary = [prev_sol_binary]
    icut_constraints = []

    for i in range(maxiter):
        t0 = time.perf_counter()
        const = create_icut_constraint(model, reaction_weights, thr, prev_sol, name='icut_'+str(i), full=full)
        model.solver.add(const)
        icut_constraints.append(const)
        try:
            prev_sol = imat(model, reaction_weights, epsilon=eps, threshold=thr, full=full)
        except:
            print('An error occured in iteration %i of icut, check if all feasible solutions have been found' % (i+1))
            break
        t1 = time.perf_counter()
        print('time for iteration '+str(i+1)+': ', t1-t0)
        if prev_sol.objective_value >= optimal_objective_value:
            all_solutions.append(prev_sol)
            prev_sol_binary = (np.abs(prev_sol.fluxes) >= thr-tol).values.astype(int)
            all_solutions_binary.append(prev_sol_binary)
        else:
            break

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    solution = EnumSolution(all_solutions, all_solutions_binary, all_solutions[0].objective_value)
    if full:
        print('full icut iterations: ', i+1)
    else:
        print('partial icut iterations: ', i+1)
    return solution

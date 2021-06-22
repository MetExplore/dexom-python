
import six
import time
import numpy as np
from sympy import sympify
from src.imat import imat
from src.result_functions import get_binary_sol
from src.enum_functions.enumeration import EnumSolution


def create_icut_constraint(model, reaction_weights, threshold, prev_sol, prev_sol_binary, name, full=False):
    """
    Creates an icut constraint on the previously found solution.
    This solution is excluded from the solution space.
    """
    if full:
        expr = sympify("1")
        newbound = sum(prev_sol_binary)
        cvector = [1 if x else -1 for x in prev_sol_binary]
        for idx, rxn in enumerate(model.reactions):
            expr += cvector[idx] * model.solver.variables["x_" + rxn.id]
    else:
        newbound = -1
        var_vals = []
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:
                y = model.solver.variables["rh_" + rid + "_pos"]
                x = model.solver.variables["rh_" + rid + "_neg"]
                if prev_sol.fluxes[rid] >= threshold:
                    var_vals.append(y-x)
                    newbound += 1
                elif prev_sol.fluxes[rid] <= -threshold:
                    var_vals.append(x - y)
                    newbound += 1
                else:
                    var_vals.append(- y - x)
            elif weight < 0:
                x = model.solver.variables["rl_" + rid]
                if np.abs(prev_sol.fluxes[rid]) < threshold:
                    var_vals.append(x)
                    newbound += 1
                else:
                    var_vals.append(- x)
        expr = sum(var_vals)
    constraint = model.solver.interface.Constraint(expr, ub=newbound, name=name)
    if expr.evalf() == 1:
        print("No reactions were found in reaction_weights when attempting to create an icut constraint")
        constraint = None
    return constraint


def icut(model, reaction_weights=None, epsilon=1e-2, threshold=1e-5, tlim=None, feas=1e-6, mipgap=1e-3, obj_tol=1e-5,
         maxiter=10, full=False):
    """
    integer-cut method

    Parameters
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    epsilon: float
        activation threshold in imat
    threshold: float
        detection threshold of activated reactions
    tlim: int
        time limit for imat
    tol: float
        tolerance for imat
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
    new_solution = imat(model, reaction_weights,
                        epsilon=epsilon, threshold=threshold, timelimit=tlim, feasibility=feas, mipgaptol=mipgap, full=full)
    new_solution_binary = get_binary_sol(new_solution, threshold)
    optimal_objective_value = new_solution.objective_value - obj_tol

    all_solutions = [new_solution]
    all_solutions_binary = [new_solution_binary]
    icut_constraints = []

    for i in range(maxiter):
        t0 = time.perf_counter()

        const = create_icut_constraint(model, reaction_weights, threshold, new_solution, new_solution_binary,
                                       name="icut_"+str(i), full=full)
        model.solver.add(const)
        icut_constraints.append(const)

        try:
            new_solution = imat(model, reaction_weights, epsilon=epsilon,
                                threshold=threshold, timelimit=tlim, feasibility=feas, mipgaptol=mipgap, full=full)
        except:
            print("An error occured in iteration %i of icut, check if all feasible solutions have been found" % (i+1))
            break
        t1 = time.perf_counter()
        print("time for iteration "+str(i+1)+": ", t1-t0)

        if new_solution.objective_value >= optimal_objective_value:
            all_solutions.append(new_solution)
            new_solution_binary = get_binary_sol(new_solution, threshold)
            all_solutions_binary.append(new_solution_binary)
        else:
            break

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    solution = EnumSolution(all_solutions, all_solutions_binary, optimal_objective_value+obj_tol)
    if full:
        print("full icut iterations: ", i+1)
    else:
        print("partial icut iterations: ", i+1)
    return solution


if __name__ == "__main__":
    from cobra.io import load_json_model
    from src.model_functions import load_reaction_weights
    model = load_json_model("recon2_2/recon2v2_corrected.json")
    reaction_weights = load_reaction_weights("recon2_2/microarray_hgnc_pval_0-01_weights.csv")

    sol = icut(model, reaction_weights)
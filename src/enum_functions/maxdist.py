

import argparse
import pandas as pd
from pathlib import Path
from cobra.io import load_json_model, load_matlab_model, read_sbml_model
from src.imat import imat, create_partial_variables, create_full_variables
from src.result_functions import read_solution, get_binary_sol, write_solution, get_obj_value_from_binary
from src.model_functions import load_reaction_weights
from src.enum_functions.enumeration import EnumSolution, get_recent_solution_and_iteration
import six
import time
from symengine import Add, sympify
from src.enum_functions.icut import create_icut_constraint


def create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol, name="maxdist_optimality", full=False):
    """
    Creates the optimality constraint for the maxdist algorithm.
    This constraint conserves the optimal objective value of the previous solution
    """
    y_variables = []
    y_weights = []
    x_variables = []
    x_weights = []

    if full:
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:
                y_pos = model.solver.variables["xf_" + rid]
                y_neg = model.solver.variables["xr_" + rid]
                y_variables.append([y_neg, y_pos])
                y_weights.append(weight)
            elif weight < 0:
                x = sympify("1") - model.solver.variables["x_" + rid]
                x_variables.append(x)
                x_weights.append(abs(weight))
    else:
        for rid, weight in six.iteritems(reaction_weights):
            if weight > 0:
                y_neg = model.solver.variables["rh_" + rid + "_neg"]
                y_pos = model.solver.variables["rh_" + rid + "_pos"]
                y_variables.append([y_neg, y_pos])
                y_weights.append(weight)
            elif weight < 0:
                x_variables.append(model.solver.variables["rl_" + rid])
                x_weights.append(abs(weight))

    lower_opt = prev_sol.objective_value - prev_sol.objective_value * obj_tol
    rh_objective = [(y[0] + y[1]) * y_weights[idx] for idx, y in enumerate(y_variables)]
    rl_objective = [x * x_weights[idx] for idx, x in enumerate(x_variables)]
    opt_const = model.solver.interface.Constraint(Add(*rh_objective) + Add(*rl_objective), lb=lower_opt, name=name)
    return opt_const


def create_maxdist_objective(model, reaction_weights, prev_sol, prev_sol_bin, only_ones=False, full=False):
    """
    Create the new objective for the maxdist algorithm.
    This objective is the minimization of similarity between the binary solution vectors
    If only_ones is set to False, the similarity will only be calculated with overlapping ones
    """
    expr = sympify("0")
    if full:
        for rxn in model.reactions:
            rid = rxn.id
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            x = model.solver.variables["x_" + rid]
            if prev_sol_bin[rid_loc] == 1:
                expr += x
            elif not only_ones:
                expr += 1 - x
    else:
        for rid, weight in six.iteritems(reaction_weights):
            rid_loc = prev_sol.fluxes.index.get_loc(rid)
            if weight > 0:
                y_neg = model.solver.variables["rh_" + rid + "_neg"]
                y_pos = model.solver.variables["rh_" + rid + "_pos"]
                if prev_sol_bin[rid_loc] == 1:
                    expr += y_neg + y_pos
                elif not only_ones:
                    expr += 1 - (y_neg + y_pos)
            elif weight < 0:
                x_rl = model.solver.variables["rl_" + rid]
                if prev_sol_bin[rid_loc] == 1:
                    expr += 1 - x_rl
                elif not only_ones:
                    expr += x_rl
    objective = model.solver.interface.Objective(expr, direction="min")
    return objective


def maxdist(model, reaction_weights, prev_sol, threshold=1e-4, obj_tol=1e-3, maxiter=10, out_path="maxdist", icut=True,
            full=False, only_ones=False):
    """
    Parameters
    maximal distance enumeration
    ----------
    model: cobrapy Model
    reaction_weights: dict
        keys = reactions and values = weights
    prev_sol: Solution instance
        a previous imat solution
    threshold: float
        detection threshold of activated reactions
    obj_tol: float
        variance allowed in the objective_values of the solutions
    maxiter: foat
        maximum number of solutions to check for
    only_ones: bool
        determines if the hamming distance is only calculated with ones, or with ones & zeros
    Returns
    -------

    """
    icut_constraints = []
    all_solutions = [prev_sol]
    prev_sol_bin = get_binary_sol(prev_sol, threshold)
    all_binary = [prev_sol_bin]

    # adding the optimality constraint: the new objective value must be equal to the previous objective value
    opt_const = create_maxdist_constraint(model, reaction_weights, prev_sol, obj_tol,
                                          name="maxdist_optimality", full=full)
    model.solver.add(opt_const)
    for i in range(maxiter):
        t0 = time.perf_counter()
        if icut:
            # adding the icut constraint to prevent the algorithm from finding the same solutions
            const = create_icut_constraint(model, reaction_weights, threshold, prev_sol, prev_sol_bin,
                                           name="icut_"+str(i), full=full)
            model.solver.add(const)
            icut_constraints.append(const)
        # defining the objective: minimize the number of overlapping ones and zeros
        objective = create_maxdist_objective(model, reaction_weights, prev_sol, prev_sol_bin, only_ones, full)
        model.objective = objective
        try:
            with model:
                prev_sol = model.optimize()
            prev_sol_bin = get_binary_sol(prev_sol, threshold)
            all_solutions.append(prev_sol)
            all_binary.append(prev_sol_bin)
        except:
            print("An error occured in iter %i of maxdis" % (i+1))
        t1 = time.perf_counter()
        print("time for iteration "+str(i+1)+": ", t1-t0)

    model.solver.remove([const for const in icut_constraints if const in model.solver.constraints])
    model.solver.remove(opt_const)
    solution = EnumSolution(all_solutions, all_binary, all_solutions[0].objective_value)
    sol = pd.DataFrame(solution.binary)
    sol.to_csv(out_path+"_solutions.csv")
    return solution


if __name__ == "__main__":
    description = "Performs the distance-maximization enumeration algorithm"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in sbml, matlab, or json format")
    parser.add_argument("-r", "--reaction_weights", default=None,
                        help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-p", "--prev_sol", default=[], help="starting solution or directory of recent solutions")
    parser.add_argument("--epsilon", type=float, default=1e-2,
                        help="Activation threshold for highly expressed reactions")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Activation threshold for all reactions")
    parser.add_argument("-t", "--timelimit", type=int, default=None, help="Solver time limit")
    parser.add_argument("-i", "--maxiter", type=int, default=10, help="Iteration limit")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver feasibility tolerance")
    parser.add_argument("--mipgap", type=float, default=1e-3, help="Solver MIP gap tolerance")
    parser.add_argument("--obj_tol", type=float, default=1e-2,
                        help="objective value tolerance, as a fraction of the original value")
    parser.add_argument("-o", "--output", default="div_enum", help="Base name of output files, without format")
    parser.add_argument("--noicut", action='store_true', help="Use this flag to remove the icut constraint")
    parser.add_argument("--full", action='store_true', help="Use this flag to assign non-zero weights to all reactions")
    args = parser.parse_args()

    fileformat = Path(args.model).suffix
    if fileformat == ".sbml" or fileformat == ".xml":
        model = read_sbml_model(args.model)
    elif fileformat == '.json':
        model = load_json_model(args.model)
    elif fileformat == ".mat":
        model = load_matlab_model(args.model)
    else:
        print("Only SBML, JSON, and Matlab formats are supported for the models")
        model = None

    try:
        model.solver = 'cplex'
    except:
        print("cplex is not available or not properly installed")

    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    a = args.dist_anneal
    if "." in args.prev_sol:
        prev_sol, prev_bin = read_solution(args.prev_sol, model, reaction_weights)
        model = create_partial_variables(model, reaction_weights, epsilon=args.epsilon)
    elif args.prev_sol:
        prev_sol, i = get_recent_solution_and_iteration(args.prev_sol, args.startsol_num)
        a = a ** i
        model = create_partial_variables(model, reaction_weights, epsilon=args.epsilon)
    else:
        prev_sol = imat(model, reaction_weights, epsilon=args.epsilon, threshold=args.threshold,
                        timelimit=args.timelimit, feasibility=args.tol, mipgaptol=args.mipgap)

    icut = True
    if args.noicut:
      icut = False

    save = False
    if args.save:
        save = True

    full = False
    if args.full:
        full = True

    model.solver.configuration.timeout = args.timelimit
    model.tolerance = args.tol
    model.solver.problem.parameters.mip.tolerances.mipgap.set(args.mipgap)
    model.solver.configuration.presolve = True

    maxdist_sol = maxdist(model=model, reaction_weights=reaction_weights, prev_sol=prev_sol, threshold=args.threshold,
                          obj_tol=args.obj_tol, maxiter=args.maxiter, out_path=args.output, icut=icut, full=args.full,
                          only_ones=False)

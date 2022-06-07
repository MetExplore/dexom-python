
import numpy as np
import pandas as pd
from pathlib import Path
from cobra import Solution
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


# def legacy_get_obj_value_from_binary(binary, reaction_weights, full_weights=True, model=None):
#     """
#     Calculates the objective value of a solution,
#     as well as the fraction of active RH and inactive RL reactions
#
#     !! The result is not always accurate !!
#     Specific case: if a RH reaction has flux below epsilon but above threshold, it will be regarded as activated,
#     but without counting for the objective value
#     """
#     if full_weights:
#         wei = list(reaction_weights.values())
#     else:
#         wei = [reaction_weights.get(rxn.id, 0) for rxn in model.reactions]
#     obj_val = sum([wei[i]*binary[i] if wei[i] >= 0 else -wei[i]*(1-binary[i]) for i in range(len(wei))])
#     max_obj = sum([abs(x) for x in reaction_weights.values()])
#     pos_wei = sum([1 for x in reaction_weights.values() if x > 0])
#     pos_true = sum([1 for i in range(len(wei)) if wei[i] > 0 and binary[i] == 1])
#     neg_wei = sum([1 for x in reaction_weights.values() if x < 0])
#     neg_true = sum([1 for i in range(len(wei)) if wei[i] < 0 and binary[i] == 0])
#     print("objective value: %.2f" % obj_val)
#     print("maximal possible value: %.2f (%.2f%s, %.2f%s)" % (max_obj, 100*pos_true/pos_wei, "% RH",
#                                                              100*neg_true/neg_wei, "% RL"))
#     return obj_val


def write_solution(model, solution, threshold, filename="imat_sol.csv"):
    """
    Writes an optimize solution as a txt file. The solution is written in a column format
    Parameters
    ----------
    solution: cobra.Solution
    threshold: float
    filename: str
    """
    tol = model.solver.configuration.tolerances.feasibility
    solution_binary = (np.abs(solution.fluxes) >= threshold-tol).values.astype(int)

    with open(filename, "w+") as file:
        file.write("reaction,fluxes,binary\n")
        for i, v in enumerate(solution.fluxes):
            file.write(solution.fluxes.index[i]+","+str(v)+","+str(solution_binary[i])+"\n")
        file.write("objective value: %f\n" % solution.objective_value)
        file.write("solver status: %s" % solution.status)
    return solution, solution_binary


def read_solution(filename, model=None, reaction_weights=None):
    binary = True
    with open(filename, "r") as f:
        reader = f.read().split("\n")
        if reader[0] == "reaction,fluxes,binary":
            binary = False
            if reader[-1] == '':
                reader.pop(-1)
            objective_value = float(reader[-2].split()[-1])
            status = reader[-1].split()[-1]
    if binary:
        fluxes = pd.read_csv(filename, index_col=0).rename(index={0: "fluxes"}).T
        fluxes.index = [rxn.id for rxn in model.reactions]
        sol_bin = list(fluxes["fluxes"])
        # objective_value = get_obj_value_from_binary(sol_bin, model, reaction_weights)
        objective_value = 0.
        status = "binary"
    else:
        df = pd.read_csv(filename, index_col=0, skipfooter=2, engine="python")
        fluxes = df["fluxes"]
        sol_bin = df["binary"].to_list()
    solution = Solution(objective_value, status, fluxes)
    return solution, sol_bin


def combine_solutions(sol_path):
    solutions = Path(sol_path).glob("*solutions.csv")
    sollist = []
    for sol in solutions:
        sollist.append(pd.read_csv(sol, index_col=0))
    fullsol = pd.concat(sollist, ignore_index=True)
    uniquesol = fullsol.drop_duplicates()
    print("There are %i unique solutions and %i duplicates." % (len(uniquesol), len(fullsol) - len(uniquesol)))
    uniquesol.to_csv(sol_path+"/combined_solutions.csv")
    return uniquesol

#
# def detect_problems(model, eps, thr, tol):
#     primals_indicators = {}
#     for key, val in model.solver.primal_values.items():
#         if "rh_"[:3] in key or "rl_" in key[:3]:
#             primals_indicators[key] = val
#
#     primals_fluxes = {}
#     for key, val in model.solver.primal_values.items():
#         if key not in primals_indicators.keys():
#             if "reverse" not in key:
#                 primals_fluxes[key] = val
#             else:
#                 newkey = "_".join(key.split("_")[:2])
#                 primals_fluxes[newkey] = val
#
#     problems = {}
#     for key, val in primals_indicators.items():
#         k = key.split("_")
#         if k[0] == "rl":
#             flux = primals_fluxes[k[1]] + primals_fluxes[k[1] + "_reverse"]
#             if val > 0.5 and flux < (thr-tol):  # uses new variable implementation
#                 problems[key] = (flux, val)
#             elif val < 0.5 and flux >= (thr-tol):  # uses new variable implementation
#                 problems[key] = (flux, val)
#         elif k[-1] == "pos":
#             flux = primals_fluxes[k[1]]
#             if val > 0.5 and flux < (eps-1000*tol):
#                 problems[key] = (flux, val)
#             elif val < 0.5 and flux >= (eps-1000*tol):
#                 problems[key] = (flux, val)
#         elif k[-1] == "neg":
#             flux = primals_fluxes[k[1] + "_reverse"]
#             if val > 0.5 and flux < (eps-1000*tol):
#                 problems[key] = (flux, val)
#             elif val < 0.5 and flux >= (eps-1000*tol):
#                 problems[key] = (flux, val)
#         else:
#             print("problem with", key)
#
#     return problems


def plot_pca(solution_path, rxn_enum_solutions=None, save_name=""):
    """
    Plots a 2-dimensional PCA of enumeration solutions

    Parameters
    ----------
    solution_path: str
        csv file of enumeration solutions
    rxn_enum_solutions: str
        csv file of enumeration solutions. If specified, will plot these solutions in a different color
    save_name: str
        name of the file to save

    Returns:
        sklearn.decomposition.PCA
    """
    X = pd.read_csv(solution_path, index_col=0)

    if rxn_enum_solutions:
        X2 = pd.read_csv(rxn_enum_solutions, index_col=0)
        X_t = pd.concat([X, X2])
    else:
        X_t = X

    pca = PCA(n_components=2)
    pca.fit(X_t)

    comp = pca.transform(X)
    x = [c[0] for c in comp]
    y = [c[1] for c in comp]

    if rxn_enum_solutions:
        comp2 = pca.transform(X2)
        x2 = [c[0] for c in comp2]
        y2 = [c[1] for c in comp2]

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=20)
    plt.ylabel('Principal Component 2', fontsize=20)
    plt.title("PCA of enumeration solutions", fontsize=20)
    if rxn_enum_solutions:
        plt.scatter(x2, y2, color="g", label="rxn-enum solutions")
    plt.scatter(x, y, color="b", label="div-enum solutions")
    plt.scatter(x[0], y[0], color="r", label="iMAT solution")
    plt.legend(fontsize="large")
    fig.savefig(save_name+"pca.png")

    return pca


if __name__ == "__main__":
    description = "Plots a 2-dimensional PCA of enumeration solutions and saves as png"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s", "--solutions", help="csv file containing diversity-enumeration solutions")
    parser.add_argument("-r", "--rxn_solutions", default=None,
                        help="(optional) csv file containing diversity-enumeration solutions")
    parser.add_argument("-o", "--out_path", default="", help="name of the file which will be saved")
    args = parser.parse_args()

    pca = plot_pca(args.solutions, rxn_enum_solutions=args.rxn_solutions, save_name=args.out_path)

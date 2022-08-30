import pandas as pd
import pytest
import pathlib
import cobra
import numpy as np
import dexom_python.model_functions as mf
import dexom_python.gpr_rules as gr
import dexom_python.imat_functions as im
import dexom_python.result_functions as rf
import dexom_python.enum_functions as enum


@pytest.fixture()
def model():
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10.json"))
    return mf.read_model(file)


@pytest.fixture()
def reaction_weights():
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_weights.csv"))
    return mf.load_reaction_weights(file)


@pytest.fixture()
def gene_weights():
    genes = pd.read_csv(str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_expression.csv")))
    genes.index = genes.pop("ID")
    return gr.expression2qualitative(genes, save=False)


@pytest.fixture()
def imatsol(model, reaction_weights):
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_imatsolution.csv"))
    solution, binary = rf.read_solution(file)
    return solution


# Testing model_functions


def test_read_model(model):
    assert type(model) == cobra.Model


def test_check_model_options(model):
    model = mf.check_model_options(model, timelimit=100, feasibility=1e-8, mipgaptol=1e-2, verbosity=3)
    assert model.solver.configuration.timeout == 100 and model.tolerance == 1e-8 and \
           model.solver.problem.parameters.mip.tolerances.mipgap.get() == 1e-2 and \
           model.solver.configuration.verbosity == 3


def test_get_all_reactions_from_model(model):
    rxn_list = mf.get_all_reactions_from_model(model, save=False)
    assert len(rxn_list) == 13


def test_get_subsystems_from_model(model):
    rxn_sub, sublist = mf.get_subsystems_from_model(model, save=False)
    assert len(rxn_sub) == 13 and len(sublist) == 0


def test_save_reaction_weights(model):
    weights = {}
    rh_reactions = ['R_a_f', 'EX_a']
    rl_reactions = ['EX_b', 'EX_e', 'EX_i']
    for r in model.reactions:
        if r.id in rh_reactions:
            weights[r.id] = 1.
        elif r.id in rl_reactions:
            weights[r.id] = -1.
        else:
            weights[r.id] = 0.
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_weights.csv"))
    reaction_weights = mf.save_reaction_weights(weights, filename=file)
    assert np.isclose(reaction_weights.values, list(weights.values())).all()


def test_load_reaction_weights(model, reaction_weights):
    assert sum([r.id in reaction_weights.keys() for r in model.reactions]) == 13


# Testing apply_gpr


def test_expression2qualitative(gene_weights):
    assert False not in (gene_weights.value_counts().sort_values().values == (5, 5, 10))


def test_apply_gpr(model, gene_weights, reaction_weights):
    weights = pd.Series(gene_weights["expr"].values, index=gene_weights.index).to_dict()
    test_wei = gr.apply_gpr(model, weights, save=False)
    assert test_wei == reaction_weights


# Testing imat


def test_create_new_partial_variables(model, reaction_weights):
    im.create_new_partial_variables(model, reaction_weights, 1, 1e-5)
    assert len(model.variables) == 41 and len(model.constraints) == 35


def test_create_full_variables(model, reaction_weights):
    im.create_full_variables(model, reaction_weights, 1, 1e-5)
    assert len(model.variables) == 65 and len(model.constraints) == 75


def test_imat(model, reaction_weights):
    solution = im.imat(model, reaction_weights, epsilon=1, threshold=1e-3)
    assert np.isclose(solution.objective_value, 4.)


def test_imat_noweights(model):
    sol = im.imat(model)
    assert type(sol) == cobra.Solution


def test_imat_fluxconsistency(model):
    weights = {r.id: 1. for r in model.reactions}
    sol = im.imat(model, weights)
    assert len(np.nonzero(sol.fluxes.values)[0]) == len(model.reactions)


def test_imat_noflux(model):
    weights = {r.id: -1. for r in model.reactions}
    sol = im.imat(model, weights)
    assert len(np.nonzero(sol.fluxes.values)[0]) == 0


# Testing result_functions


def test_read_solution(imatsol):
    assert np.isclose(imatsol.objective_value, 4.) and len(imatsol.fluxes) == 13


def test_write_solution(model, imatsol):
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_imatsolution.csv"))
    solution, binary = rf.write_solution(model, imatsol, threshold=1e-3, filename=file)
    assert len(binary) == len(solution.fluxes)


# Testing enumeration functions


def test_rxn_enum(model, reaction_weights, imatsol):
    rxn_sol = enum.rxn_enum(model=model, reaction_weights=reaction_weights, prev_sol=imatsol)
    assert np.isclose(rxn_sol.objective_value, 4.) and len(rxn_sol.solutions) == 3


def test_icut_partial(model, reaction_weights, imatsol):
    icut_sol = enum.icut(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=10, full=False)
    assert np.isclose(icut_sol.objective_value, 4.) and len(icut_sol.solutions) == 3


def test_icut_full(model, reaction_weights, imatsol):
    icut_sol = enum.icut(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=10, full=True)
    assert np.isclose(icut_sol.objective_value, 4.) and len(icut_sol.solutions) == 3


def test_maxdist_partial(model, reaction_weights, imatsol):
    maxdist_sol = enum.maxdist(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=4, full=False,)
    assert np.isclose(maxdist_sol.objective_value, 4.) and len(maxdist_sol.solutions) == 3


def test_maxdist_full(model, reaction_weights, imatsol):
    maxdist_sol = enum.maxdist(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=4, full=True,)
    assert np.isclose(maxdist_sol.objective_value, 4.) and len(maxdist_sol.solutions) == 3


def test_diversity_enum_partial(model, reaction_weights, imatsol):
    div_enum_sol, div_enum_res = enum.diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=imatsol,
                                                     maxiter=4, full=False)
    assert np.isclose(div_enum_sol.objective_value, 4.) and len(div_enum_sol.solutions) == 3


def test_diversity_enum_full(model, reaction_weights, imatsol):
    div_enum_sol, div_enum_res = enum.diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=imatsol,
                                                     maxiter=4, full=True)
    assert np.isclose(div_enum_sol.objective_value, 4.) and len(div_enum_sol.solutions) == 3


def test_plot_pca():
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_rxnenum_solutions.csv"))
    pca = rf.plot_pca(file, save=False)
    assert np.shape(pca.components_) == (2, 13)

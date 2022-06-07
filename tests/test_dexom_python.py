
import pytest
import pathlib
import cobra
import numpy as np
import dexom_python.model_functions as mf
import dexom_python.imat as im
import dexom_python.result_functions as rf


@pytest.fixture()
def model():
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10.json"))
    return mf.read_model(file)


@pytest.fixture()
def reaction_weights():
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_weights.csv"))
    return mf.load_reaction_weights(file)


@pytest.fixture()
def imatsol(model, reaction_weights):
    return im.imat(model, reaction_weights, epsilon=1, threshold=1e-3)


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
    assert len(rxn_sub)==13 and len(sublist)==0


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
    assert sum([r.id in reaction_weights.keys() for r in model.reactions])==13


# Testing imat


def test_create_partial_variables(model, reaction_weights):
    im.create_partial_variables(model, reaction_weights, 1)
    assert len(model.variables) == 33 and len(model.constraints) == 20


def test_create_new_partial_variables(model, reaction_weights):
    im.create_new_partial_variables(model, reaction_weights, 1, 1e-5)
    assert len(model.variables) == 41 and len(model.constraints) == 35


def test_create_full_variables(model, reaction_weights):
    im.create_full_variables(model, reaction_weights, 1, 1e-5)
    assert len(model.variables) == 65 and len(model.constraints) == 75


def test_imat(imatsol):
    assert np.isclose(imatsol.objective_value, 4.)


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


def test_write_solution(model, imatsol):
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_imatsolution.csv"))
    solution, binary = rf.write_solution(model, imatsol, threshold=1e-3, filename=file)
    assert len(binary) == len(solution.fluxes)


def test_read_solution(model):
    file = str(pathlib.Path(__file__).parent.joinpath("model", "example_r13m10_imatsolution.csv"))
    solution, binary = rf.read_solution(file)
    assert len(binary) == len(solution.fluxes)

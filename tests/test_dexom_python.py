import pandas as pd
import pytest
import pathlib
import cobra
import argparse
import numpy as np
import dexom_python.model_functions as mf
import dexom_python.gpr_rules as gr
import dexom_python.imat_functions as im
import dexom_python.result_functions as rf
import dexom_python.pathway_enrichment as pe
import dexom_python.enum_functions as enum
from dexom_python.model_functions import DEFAULT_VALUES as DV
from unittest import mock


# Define some global variables used across multiple tests

GLOB_modelstring = str(pathlib.Path(__file__).parent.joinpath('model', 'example_r13m10.json'))
GLOB_weightstring = str(pathlib.Path(__file__).parent.joinpath('model', 'example_r13m10_weights.csv'))
GLOB_expressionstring = str(pathlib.Path(__file__).parent.joinpath('model', 'example_r13m10_expression.csv'))
GLOB_expressiondfstring = str(pathlib.Path(__file__).parent.joinpath('model', 'example_r13m10_expression_gprtest.csv'))
GLOB_imatstring = str(pathlib.Path(__file__).parent.joinpath('model', 'results', 'example_r13m10_imatsolution.csv'))
GLOB_rxnsols = str(pathlib.Path(__file__).parent.joinpath('model', 'results', 'example_r13m10_rxnenum_solutions.csv'))


@pytest.fixture()
def model():
    return mf.read_model(modelfile=GLOB_modelstring)


@pytest.fixture()
def reaction_weights():
    return mf.load_reaction_weights(filename=GLOB_weightstring)


@pytest.fixture()
def gene_weights():
    genes = pd.read_csv(GLOB_expressionstring)
    genes.index = genes.pop('ID')
    return gr.expression2qualitative(genes=genes, save=False)


@pytest.fixture()
def imatsol(model, reaction_weights):
    solution, binary = rf.read_solution(GLOB_imatstring)
    return solution


# Testing model_functions


def test_read_model_json(model):
    assert type(model) == cobra.Model


def test_read_model_sbml():
    m = mf.read_model(modelfile=str(pathlib.Path(__file__).parent.joinpath('model', 'example_r13m10.xml')))
    assert type(m) == cobra.Model


def test_read_model_mat():
    m = mf.read_model(modelfile=str(pathlib.Path(__file__).parent.joinpath('model', 'example_r13m10.mat')))
    assert type(m) == cobra.Model


def test_check_model_options(model):
    model = mf.check_model_options(model=model, timelimit=100, tolerance=1e-8, mipgaptol=1e-2, verbosity=3)
    assert model.solver.configuration.timeout == 100 and model.tolerance == 1e-8 and \
           model.solver.configuration.verbosity == 3


def test_get_all_reactions_from_model(model):
    rxn_list = mf.get_all_reactions_from_model(model=model, save=False)
    assert len(rxn_list) == 13


def test_get_subsystems_from_model(model):
    rxn_sub, sublist = mf.get_subsystems_from_model(model=model, save=False)
    assert len(rxn_sub) == 13 and len(sublist) == 3


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
    reaction_weights = mf.save_reaction_weights(reaction_weights=weights, filename=GLOB_weightstring)
    assert np.isclose(reaction_weights.values, list(weights.values())).all()


def test_load_reaction_weights(model, reaction_weights):
    assert sum([r.id in reaction_weights.keys() for r in model.reactions]) == 13


def test_check_threshold_tolerance(model):
    assert mf.check_threshold_tolerance(model, epsilon=3e-4, threshold=2e-4) == 0


def test_check_threshold_tolerance_thresholderror(model):
    with pytest.raises(ValueError, match=r'The threshold parameter value') as e:
        mf.check_threshold_tolerance(model, epsilon=3e-4, threshold=1e-4)
    assert e.match(r'0.0001')


def test_check_threshold_tolerance_epsilonerror(model):
    with pytest.raises(ValueError, match=r'The epsilon parameter value') as e:
        mf.check_threshold_tolerance(model, epsilon=2e-4, threshold=2e-4)
    assert e.match(r'0.0002')


# Testing gpr_rules


def test_expression2qualitative(gene_weights):
    assert False not in (gene_weights.value_counts().sort_values().values == (5, 5, 10))


def test_apply_gpr(model, gene_weights, reaction_weights):
    weights = pd.Series(gene_weights['expr'].values, index=gene_weights.index).to_dict()
    test_wei = gr.apply_gpr(model=model, gene_weights=weights, save=False)
    assert test_wei == reaction_weights


# Testing imat_functions


def test_create_new_partial_variables(model, reaction_weights):
    im.create_new_partial_variables(model=model, reaction_weights=reaction_weights, epsilon=DV['epsilon'],
                                    threshold=DV['threshold'])
    assert len(model.variables) == 41 and len(model.constraints) == 25


def test_create_full_variables(model, reaction_weights):
    im.create_full_variables(model=model, reaction_weights=reaction_weights, epsilon=DV['epsilon'],
                             threshold=DV['threshold'])
    assert len(model.variables) == 65 and len(model.constraints) == 49


def test_imat(model, reaction_weights):
    solution = im.imat(model=model, reaction_weights=reaction_weights, epsilon=DV['epsilon'], threshold=DV['threshold'])
    assert np.isclose(solution.objective_value, 4.)


def test_imat_noweights(model):
    sol = im.imat(model=model)
    assert type(sol) == cobra.Solution


def test_imat_fluxconsistency(model):
    weights = {r.id: 1. for r in model.reactions}
    sol = im.imat(model=model, reaction_weights=weights)
    assert len(np.nonzero(sol.fluxes.values)[0]) == len(model.reactions)


def test_imat_noflux(model):
    weights = {r.id: -1. for r in model.reactions}
    sol = im.imat(model=model, reaction_weights=weights)
    assert len(np.nonzero(sol.fluxes.values)[0]) == 0


# Testing result_functions


def test_read_solution(imatsol):
    assert np.isclose(imatsol.objective_value, 4.) and len(imatsol.fluxes) == 13


def test_write_solution(model, imatsol):
    solution, binary = rf.write_solution(model=model, solution=imatsol, threshold=DV['threshold'],
                                         filename=GLOB_imatstring)
    assert len(binary) == len(solution.fluxes)


# Testing enumeration functions

def test_rxn_enum(model, reaction_weights, imatsol):
    rxn_sol = enum.rxn_enum(model=model, reaction_weights=reaction_weights, prev_sol=imatsol)
    assert np.isclose(rxn_sol.objective_value, 4.) and len(rxn_sol.unique_solutions) == 3


def test_icut_partial(model, reaction_weights, imatsol):
    icut_sol = enum.icut(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=10, full=False)
    assert np.isclose(icut_sol.objective_value, 4.) and len(icut_sol.solutions) == 3


def test_icut_full(model, reaction_weights, imatsol):
    icut_sol = enum.icut(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=10, full=True)
    assert np.isclose(icut_sol.objective_value, 4.) and len(icut_sol.solutions) == 3


def test_maxdist_partial(model, reaction_weights, imatsol):
    maxdist_sol = enum.maxdist(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=4, full=False)
    assert np.isclose(maxdist_sol.objective_value, 4.) and len(maxdist_sol.solutions) == 3


def test_maxdist_full(model, reaction_weights, imatsol):
    maxdist_sol = enum.maxdist(model=model, reaction_weights=reaction_weights, prev_sol=imatsol, maxiter=4, full=True)
    assert np.isclose(maxdist_sol.objective_value, 4.) and len(maxdist_sol.solutions) == 3


def test_diversity_enum_partial(model, reaction_weights, imatsol):
    div_enum_sol, div_enum_res = enum.diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=imatsol,
                                                     maxiter=4, full=False)
    assert np.isclose(div_enum_sol.objective_value, 4.) and len(div_enum_sol.solutions) == 3


def test_diversity_enum_full(model, reaction_weights, imatsol):
    div_enum_sol, div_enum_res = enum.diversity_enum(model=model, reaction_weights=reaction_weights, prev_sol=imatsol,
                                                     maxiter=4, full=True)
    assert np.isclose(div_enum_sol.objective_value, 4.) and len(div_enum_sol.solutions) == 3


# Testing enumeration helper functions


def test_plot_pca():
    pca = rf.plot_pca(GLOB_rxnsols, save=False)
    assert np.shape(pca.components_) == (2, 13)


def test_read_prev_sol_imat(model, reaction_weights):
    sol, _ = enum.read_prev_sol(GLOB_imatstring, model, reaction_weights)
    assert np.isclose(sol.objective_value, 4.)


def test_read_prev_sol_directory(model, reaction_weights):
    file = str(pathlib.Path(__file__).parent.joinpath('model', 'results'))
    sol, a = enum.read_prev_sol(file, model, reaction_weights, pattern='*solution*.csv')
    assert np.isclose(sol.objective_value, 4.)


def test_read_prev_sol_binary(model, reaction_weights):
    sol, _ = enum.read_prev_sol(GLOB_rxnsols, model, reaction_weights)
    assert np.isclose(sol.objective_value, 4.)


def test_check_reaction_weights():
    try:
        enum.enumeration.check_reaction_weights({})
        raised1 = False
    except ValueError:
        raised1 = True
    try:
        enum.enumeration.check_reaction_weights({'a': 0, 'b': 0})
        raised2 = False
    except ValueError:
        raised2 = True
    assert raised1 and raised2


# Testing permutation


def test_permute_genelabels(model, gene_weights):
    perm_sol, perm_bin, perm_recs, perm_genes = enum.permute_genelabels(model=model, allgenes=gene_weights['expr'],
                                                                        nperms=4)
    assert len(perm_sol) == 4 and len(perm_bin) == 4 and perm_recs.shape == (13, 4) and perm_genes.shape == (20, 4)


# Testing main functions

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, gene_ID='ID', gene_score='expr',
                                            gene_file=GLOB_expressionstring, duplicates='remove',
                                            convert=True, quantiles='0.25', null=0., significant='both',
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'example_r13m10_weights'))))
def test_gpr_main(mock_args):
    res = gr._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, gene_ID='ID', gene_score=None,
                                            gene_file=GLOB_expressiondfstring, duplicates='remove',
                                            convert=False, threshold='0.25', null=0., significant='both',
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'example_r13m10_weights_df'))))
def test_gpr_main_dataframe(mock_args):
    res = gr._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, epsilon=DV['epsilon'],
                                            reaction_weights=GLOB_weightstring, tol=DV['tolerance'],
                                            threshold=DV['threshold'], timelimit=DV['timelimit'], mipgap=DV['mipgap'],
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_imatsolution'))))
def test_imat_main(mock_args):
    res = im._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(solutions=GLOB_rxnsols, rxn_solutions=None,
                                            out_path=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_'))))
def test_result_functions_main(mock_args):
    res = rf._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, epsilon=DV['epsilon'],
                                            reaction_weights=GLOB_weightstring, tol=DV['tolerance'],
                                            threshold=DV['threshold'], timelimit=DV['timelimit'], mipgap=DV['mipgap'],
                                            obj_tol=DV['obj_tol'], save=False, prev_sol=GLOB_imatstring,
                                            reaction_list=None, range='_',
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_rxnenum'))))
def test_rxnenum_main(mock_args):
    res = enum.rxn_enum_functions._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, epsilon=DV['epsilon'],
                                            reaction_weights=GLOB_weightstring, tol=DV['tolerance'],
                                            threshold=DV['threshold'], timelimit=DV['timelimit'], mipgap=DV['mipgap'],
                                            obj_tol=DV['obj_tol'], save=False, prev_sol=GLOB_imatstring, full=False,
                                            maxiter=DV['maxiter'],
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_icut'))))
def test_icut_main(mock_args):
    res = enum.icut_functions._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, epsilon=DV['epsilon'],
                                            reaction_weights=GLOB_weightstring, tol=DV['tolerance'],
                                            threshold=DV['threshold'], timelimit=DV['timelimit'], mipgap=DV['mipgap'],
                                            obj_tol=DV['obj_tol'], save=False, prev_sol=GLOB_imatstring, noicut=False,
                                            full=False, maxiter=4, onlyones=False,
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_maxdist'))))
def test_maxdist_main(mock_args):
    res = enum.maxdist_functions._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, epsilon=DV['epsilon'],
                                            reaction_weights=GLOB_weightstring, tol=DV['tolerance'],
                                            threshold=DV['threshold'], timelimit=DV['timelimit'], mipgap=DV['mipgap'],
                                            obj_tol=DV['obj_tol'], save=False, prev_sol=GLOB_imatstring, noicut=False,
                                            full=False, maxiter=4, dist_anneal=DV['dist_anneal'],
                                            startsol=1,
                                            output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_divenum'))))
def test_divenum_main(mock_args):
    res = enum.diversity_enum_functions._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(solutions=GLOB_rxnsols,
                                            model=GLOB_modelstring, sublist=None, subframe=None,
                                            out_path=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_'))))
def test_pathway_main_json(mock_args):
    res = pe._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(solutions=GLOB_rxnsols,
                                            model=GLOB_modelstring, sublist=None, subframe=None,
                                            out_path=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_'))))
def test_pathway_main_sbml(mock_args):
    res = pe._main()
    assert res is True


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(model=GLOB_modelstring, gene_file=GLOB_expressionstring, npermutations=4,
                                            gene_index='false', output=str(pathlib.Path(__file__).parent.joinpath(
                                                'model', 'results', 'example_r13m10_perms_')), error_tol=10))
def test_permutation_main(mock_args):
    res = enum.permutation_functions._main()
    assert res is True
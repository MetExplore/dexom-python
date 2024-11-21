import optlang
import pandas as pd
import numpy as np
import cobra
from pathlib import Path
from cobra.io import load_json_model, read_sbml_model, load_matlab_model
from cobra.flux_analysis import find_blocked_reactions
from warnings import warn
from cobra.exceptions import SolverNotFound, OptimizationError
from dexom_python.default_parameter_values import DEFAULT_VALUES


def read_model(modelfile, solver='cplex'):
    config = cobra.Configuration()
    try:
        config.solver=solver
    except SolverNotFound:
        warn('The solver: %s is not available or not properly installed\n' % solver)
    fileformat = Path(modelfile).suffix
    model = None
    if fileformat == '.sbml' or fileformat == '.xml':
        model = read_sbml_model(modelfile)
    elif fileformat == '.json':
        model = load_json_model(modelfile)
    elif fileformat == '.mat':
        model = load_matlab_model(modelfile)
    elif fileformat == '':
        warn('Wrong model path')
    else:
        raise TypeError('Only SBML, JSON, and Matlab formats are supported for the models')
    try:
        model.solver = solver
    except SolverNotFound:
        warn('The solver: %s is not available or not properly installed\n' % solver)
    return model


def check_model_options(model, timelimit=DEFAULT_VALUES['timelimit'], feasibility=DEFAULT_VALUES['tolerance'],
                        mipgaptol=DEFAULT_VALUES['mipgap'], verbosity=DEFAULT_VALUES['verbosity']):
    model.solver.configuration.timeout = timelimit
    model.tolerance = feasibility
    model.solver.configuration.verbosity = verbosity
    model.solver.configuration.presolve = True
    if hasattr(optlang, 'cplex_interface'):
        if isinstance(model.solver, optlang.cplex_interface.Model):
            model.solver.problem.parameters.mip.tolerances.mipgap.set(mipgaptol)
    else:
        warn('setting the MIP gap tolerance is only available with the cplex solver')
    return model


def check_threshold_tolerance(model, epsilon, threshold):
    cobra_config = cobra.Configuration()
    limit = model.tolerance * max(abs(cobra_config.upper_bound), abs(cobra_config.lower_bound))
    if threshold < 2*limit:
        raise ValueError('The threshold parameter value is too low compared to the current model tolerance. '
                         'Current threshold value: %s. Current tolerance value:%s. Minimum threshold value: %s'
                         % (str(threshold), str(model.tolerance), str(2*limit)))
    if epsilon < np.around(threshold + limit, 10):
        raise ValueError('The epsilon parameter value is too low compared to the current threshold and model tolerance.'
                         ' Current epsilon value: %s. Current threshold value: %s. Current tolerance value:%s. '
                         'Minimum epsilon value: %s'
                         % (str(epsilon), str(threshold), str(model.tolerance), str(threshold + limit)))
    return 0


def check_constraint_primal_values(model):
    for c in model.constraints:
        if c.ub is None:
            if c.primal < c.lb - model.tolerance:
                print('Invalid constraint value for %s: (lb, primal, ub) = (%s, %s, %s)'
                      % (c.name, str(c.lb), str(c.primal), str(c.ub)))
                raise OptimizationError('Invalid constraint value for %s ' % c.name)
        elif c.lb is None:
            if c.primal > c.ub + model.tolerance:
                print('Invalid constraint value for %s: (lb, primal, ub) = (%s, %s, %s)'
                      % (c.name, str(c.lb), str(c.primal), str(c.ub)))
                raise OptimizationError('Invalid constraint value for  %s ' % c.name)
        elif c.primal > c.ub + model.tolerance or c.primal < c.lb - model.tolerance:
            print('Invalid constraint value for %s: (lb, primal, ub) = (%s, %s, %s)'
                  % (c.name, str(c.lb), str(c.primal), str(c.ub)))
            raise OptimizationError('Invalid constraint value for  %s ' % c.name)
    return True


def get_all_reactions_from_model(model, save=True, shuffle=True, out_path=''):
    """
    Outputs a list of all reactions in the model. If possible, all blocked reactions are removed.
    Optionally, the reaction-list can be shuffled.

    Parameters
    ----------
    model: cobra.Model

    save: bool
        by default, exports the reactions in a csv format
    shuffle: bool
        set to True to shuffle the order of the reactions
    out_path: str
        output path
    Returns
    -------
    rxn_list: A list of all reactions in the model
    """
    rxn_list = [r.id for r in model.reactions]
    try:
        if hasattr(model, "_sbml"):
            model._sbml['created'] = None
            # In level 3 SMBL models, the model._sbml['created'] attribute is a SwigPyObject
            # which causes an exception in cobra.flux_analysis.find_blocked_reactions
        blocked = find_blocked_reactions(model)
        rxn_list = list(set(rxn_list) - set(blocked))
    except:
        warn("Could not find blocked reactions. Output list contains all reactions of the model.")
    if save:
        pd.Series(rxn_list).to_csv(out_path + model.id + '_reactions.csv', header=False, index=False)
    if shuffle:
        np.random.shuffle(rxn_list)
        if save:
            pd.Series(rxn_list).to_csv(out_path + model.id + '_reactions_shuffled.csv', header=False, index=False)
    return rxn_list


def get_subsystems_from_model(model, save=True, out_path=''):
    """
    Creates a list of all subsystems of a model and their associated reactions

    Parameters
    ----------
    model: cobra.Model
    save: bool
    out_path: str

    Returns
    -------
    rxn_sub: pandas.DataFrame
        a DataFrame with reaction names as index and subsystem name as column
    sub_list: list
        a list of subsystems
    """

    rxn_sub = {}
    sub_list = []
    i = 0
    for rxn in model.reactions:
        rxn_sub[i] = (rxn.id, rxn.subsystem)
        i += 1
        if rxn.subsystem not in sub_list:
            sub_list.append(rxn.subsystem)
    if sub_list[-1] == '':
        sub_list.pop()
    sub_list.sort()
    rxn_sub = pd.DataFrame.from_dict(rxn_sub, orient='index', columns=['ID', 'subsystem'])
    if save:
        rxn_sub.to_csv(out_path+model.id+'_reactions_subsystems.csv')
        with open(out_path+model.id+'_subsystems_list.txt', 'w+') as file:
            file.write(';'.join(sub_list))
    return rxn_sub, sub_list


def save_reaction_weights(reaction_weights, filename='reaction_weights.csv'):
    """
    Parameters
    ----------
    reaction_weights: dict
        a dictionary where keys = reaction IDs and values = weights
    filename: str

    Returns
    -------
    reaction_weights: pandas.DataFrame
    """
    df = pd.DataFrame(reaction_weights.items(), columns=['reactions', 'weights'])
    df.to_csv(filename)
    df.index = df['reactions']
    return df['weights']


def load_reaction_weights(filename, rxn_names='reactions', weight_names='weights'):
    """
    loads reaction weights from a .csv file

    Parameters
    ----------
    filename: str
        the path + name of a .csv file containing reaction weights
    rxn_names: str
        the name of the column containing the reaction names
    weight_names: str
        the name of the column containing the weights

    Returns
    -------
    reaction_weights: dict
    """
    df = pd.read_csv(filename)
    df.index = df[rxn_names]
    reaction_weights = df[weight_names].to_dict()
    return {str(k): float(v) for k, v in reaction_weights.items() if float(v) == float(v)}

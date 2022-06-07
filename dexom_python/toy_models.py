
from cobra import Model, Reaction, Metabolite
from cobra.io import save_json_model
from dexom_python.model_functions import save_reaction_weights


def create_reaction(model, rname, formula, gene_rule=None, fullname=None, lower_bound=0., upper_bound=1000.):

    rxn = Reaction(rname)
    model.add_reactions([rxn])
    rxn.name = rname if fullname is None else fullname
    rxn.gene_reaction_rule = rname if gene_rule is None else gene_rule
    rxn.add_metabolites(formula)
    rxn.bounds = (lower_bound, upper_bound)

    return model


def small4M(export=False, solver='cplex'):
    """
    creates the small4M model

    Parameters
    ----------
    export: bool
        if True, exports the model as .json and the reaction weights as .csv
    solver: str
        a valid cobrapy solver

    Returns
    -------
    model: cobra.Model
    reaction_weights: dict
    """
    model = Model(id_or_model="small4M", name="small4M_python")
    model.solver = solver
    metabolite_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X', 'Y', 'Z']
    metabolites = [Metabolite(m) for m in metabolite_names]
    model.add_metabolites(metabolites)

    reaction_names = ['EX_A', 'EX_D', 'EX_X', 'EX_C', 'EX_Z', 'EX_G', 'EX_Y', 'EX_T', 'RAB', 'RBC', 'RFG', 'RCF',
                      'RDE', 'REF1', 'REF2', 'REF3', 'RBE']
    reaction_formulas = [
        {"A": 1.},
        {"D": 1.},
        {"X": 1.},
        {"C": 1.},
        {"Z": 1.},
        {"G": -1.},
        {"Y": -1.},
        {"T": -1.},
        {"A": -1., "B": 1.},
        {"B": -1., "C": 1.},
        {"F": -1., "G": 1.},
        {"C": -1., "F": 1.},
        {"D": -1., "E": 1.},
        {"E": -1., "X": -1., "F": 1., "Y": 1.},
        {"E": -1., "F": 1.},
        {"E": -1., "Z": -1., "F": 1., "T": 1.},
        {"B": -1., "E": 1.},
    ]

    for idx, react_name in enumerate(reaction_names):
        create_reaction(model, react_name, reaction_formulas[idx])

    reversible_reactions = ['RCF', 'RBE']  # deciding which reactions should be reversible
    # making RBC reversible allows to obtain unique optimal solution with adequacy of 4
    for react_name in reversible_reactions:
        model.reactions.get_by_id(react_name).lower_bound = -1000.

    # create reaction weights
    reaction_weights = {}
    rh_reactions = ['RFG']
    rl_reactions = ['RAB', 'RDE', 'RCF']

    for rname in reaction_names:
        if rname in rh_reactions:
            reaction_weights[rname] = 1.
        elif rname in rl_reactions:
            reaction_weights[rname] = -1.

    if export:
        save_json_model(model, "small4M.json")
        save_reaction_weights(reaction_weights, "small4M_weights.csv")

    return model, reaction_weights


def small4S(export=False, solver='cplex'):
    """
    creates the small4S model

    Parameters
    ----------
    export: bool
        if True, exports the model as .json and the reaction weights as .csv
    solver: str
        a valid cobrapy solver

    Returns
    -------
    model: cobra.Model
    reaction_weights: dict
    """

    model = Model('small4S_python')
    model.solver = solver

    metabolite_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    metabolites = [Metabolite(m) for m in metabolite_names]
    model.add_metabolites(metabolites)

    reaction_names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']
    reaction_formulas = [
        {"A": -1., "B": 1.},
        {"B": -1., "C": 1.},
        {"C": -1., "D": 1.},
        {"D": -1., "H": 1.},
        {"A": -1., "E": 1.},
        {"E": -1., "F": 1.},
        {"F": -1., "G": 1.},
        {"G": -1., "H": 1.},
        {"A": -1., "H": 1.},
        {"A": 2.},
        {"H": -1.},
    ]

    for idx, react_name in enumerate(reaction_names):
        create_reaction(model, react_name, reaction_formulas[idx])

    # create reaction weights
    reaction_weights = {}
    rh_reactions = ['R2', 'R6', 'R9']
    rl_reactions = ['R3', 'R7']

    for rname in reaction_names:
        if rname in rh_reactions:
            reaction_weights[rname] = 1.
        elif rname in rl_reactions:
            reaction_weights[rname] = -1.

    if export:
        save_json_model(model, "small4S.json")
        save_reaction_weights(reaction_weights, "small4S_weights.csv")

    return model, reaction_weights


def dagNet(num_layers, num_metabolites_per_layer, export=False, solver='cplex'):
    """
    Creates a dagNet model where the metabolites of successive layers are all interconnected
    Parameters
    ----------
    num_layers: int
        number of layers
    num_metabolites_per_layer: int
        number of metabolites per layer
    export: bool
        if True, exports the model as .json and the reaction weights as .csv
    solver: str
        a valid cobrapy solver

    Returns
    -------
    model: cobra.Model
    reaction_weights: dict
    """

    model = Model("dagNet_python")
    model.solver = solver
    nl = num_layers + 2

    for i in range(nl):
        if i == 0:
            model.add_metabolites([Metabolite('Met00')])
            model.add_metabolites([Metabolite('MetSINK')])
            create_reaction(model, 'R_in', {'Met00': 1.}, lower_bound=1.)

        elif i == nl-1:
            for j, prev_met in enumerate(layer_mets):
                react_name = 'R_'+prev_met+'_SINK'
                formula = {prev_met: -1., 'MetSINK': 1.}
                create_reaction(model, react_name, formula)

        else:
            layer_mets = []
            for j in range(num_metabolites_per_layer):
                met_name = 'Met'+str(i)+str(j)
                model.add_metabolites([Metabolite(met_name)])
                layer_mets.append(met_name)
                num_mets = 1 if i == 1 else num_metabolites_per_layer
                for k in range(num_mets):
                    prev_met = 'Met'+str(i-1)+str(k)
                    react_name = 'R_'+prev_met+'_'+met_name
                    formula = {prev_met: -1., met_name: 1.}
                    create_reaction(model, react_name, formula)

    create_reaction(model, 'R_out', {'MetSINK': -1.}, lower_bound=1.)

    # create reaction weights
    # for this simple example, all reactions are lowly expressed
    reaction_weights = {}
    for rec in model.reactions:
        reaction_weights[rec.name] = -1.

    if export:
        save_json_model(model, "dagNet"+str(num_layers)+str(num_metabolites_per_layer)+".json")
        save_reaction_weights(reaction_weights, 'dagNet'+str(num_layers)+str(num_metabolites_per_layer)+'_weights.csv')

    return model, reaction_weights


def r13m10(export=False, solver='cplex'):
    model = Model(id_or_model="r13m10", name="r13m10_cobra")
    model.solver = solver
    metabolite_names = ['a', 'b', 'e', 'g', 'h', 'i', 'j', 'f', 'c', 'd']
    metabolites = [Metabolite(m) for m in metabolite_names]
    model.add_metabolites(metabolites)

    reaction_names = ['EX_a', 'EX_b', 'EX_e', 'EX_g', 'EX_h', 'EX_i', 'EX_j', 'R_a_f', 'R_ab_cd', 'R_ef_gh', 'R_f_i',
                      'R_cj_i', 'R_d_j']
    reaction_formulas = [
         {'a': -1.},
         {'b': -1.},
         {'e': -1.},
         {'g': -1.},
         {'h': -1.},
         {'i': -1.},
         {'j': -1.},
         {'a': -2., 'f': 1.},
         {'a': -1., 'b': -1., 'c': 2., 'd': 3.},
         {'e': -1., 'f': -1., 'g': 1., 'h': 5.},
         {'f': -1., 'i': 1.},
         {'c': -2., 'j': -1., 'i': 3.},
         {'d': -1., 'j': 1.}]

    reaction_bounds = [
        (-10., 50.),
        (-10., 50.),
        (-10., 50.),
        (-50., 50.),
        (-50., 50.),
        (-20., 50.),
        (-50., 50.),
        (-50., 50.),
        (0., 50.),
        (0., 100.),
        (-100., 100.),
        (0., 100.),
        (-10., 10.),
    ]
    gene_rules = ['G_IMP_A', 'G_IMP_B', 'G_IMP_E', 'G_IMP_G', 'G_IMP_H', 'G_IMP_I', 'G_IMP_J', 'G_A_F', 'G_AB_CD',
                  'G_EF_GH', 'G_F_I', 'G_CJ_I', 'G_D_J']

    for idx, react_name in enumerate(reaction_names):
        create_reaction(model, react_name, reaction_formulas[idx], gene_rule=gene_rules[idx],
                        lower_bound=reaction_bounds[idx][0], upper_bound=reaction_bounds[idx][1])

    # create reaction weights
    reaction_weights = {}
    rh_reactions = ['EX_a', 'R_a_f']
    rl_reactions = ['EX_b', 'EX_e', 'EX_i']

    for rname in reaction_names:
        if rname in rh_reactions:
            reaction_weights[rname] = 1.
        elif rname in rl_reactions:
            reaction_weights[rname] = -1.
        else:
            reaction_weights[rname] = 0.

    if export:
        save_json_model(model, "tests/model/example_r13m10.json")
        save_reaction_weights(reaction_weights, "tests/model/example_r13m10_weights.csv")
    return model, reaction_weights


if __name__ == '__main__':
    m = r13m10(export=True)

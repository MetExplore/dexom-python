
from cobra import Model, Reaction, Metabolite


def create_reaction(model, rname, formula, gene_rule=None, fullname=None, lower_bound=0., upper_bound=1000.):

    rxn = Reaction(rname)
    model.add_reactions([rxn])
    rxn.name = rname if fullname is None else fullname
    rxn.gene_reaction_rule = rname if gene_rule is None else gene_rule
    rxn.add_metabolites(formula)
    rxn.bounds = (lower_bound,upper_bound)

    return model


def small4M(solver='cplex'):
    """
    Creates the small4M example model
    returns a cobra.Model instance
    """
    print("in small4M, before Model call")
    model = Model('small4M_python')
    print("in small4M, before solver call")
    model.solver = solver
    print("in small4M, before metabolites")
    metabolite_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X', 'Y', 'Z']
    metabolites = [Metabolite(m) for m in metabolite_names]
    model.add_metabolites(metabolites)
    print("in small4M, before reactions")
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
    RH_reactions = ['RFG']
    RL_reactions = ['RAB', 'RDE', 'RCF']

    for rname in reaction_names:
        if rname in RH_reactions:
            reaction_weights[rname] = 1.
        elif rname in RL_reactions:
            reaction_weights[rname] = -1.

    return model, reaction_weights


def small4S(solver='cplex'):
    """
    Creates the small4S example model
    returns a cobra.Model instance
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
    RH_reactions = ['R2', 'R6', 'R9']
    RL_reactions = ['R3', 'R7']

    for rname in reaction_names:
        if rname in RH_reactions:
            reaction_weights[rname] = 1.
        elif rname in RL_reactions:
            reaction_weights[rname] = -1.

    return model, reaction_weights


def dagNet(num_layers, num_metabolites_per_layer, solver='cplex'):
    """
    Creates a dagNet model where the metabolites of successive layers are all interconnected
    :param num_layers: umber of layers
    :param num_metabolites_per_layer: number of metabolites per layer
    :param solver: a valid solver for cobrapy
    :return: cobra.Model instance
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

    return model, reaction_weights

import ruamel.yaml as yaml
import os
import dexom_python as dp

# read configuration from YAML files
yaml_reader = yaml.YAML(typ='safe')
with open('dexom_python/cluster_utils/cluster_config.yaml', 'r') as file:
    temp = file.read()
config = yaml_reader.load(temp)

if config['output_path']:
    outpath = config['output_path']
    os.makedirs(outpath, exist_ok=True)
    if outpath[-1] not in ['/', '\\']:
        outpath += '/'
else:
    outpath = ''

a = config['approach']

outputs = {
    'grouped': outpath + 'grouped.txt',
    'separate': outpath + 'separate.txt',
    'rxn': outpath + 'rxn.txt',
    'div': outpath + 'div.txt',
    'permutation': outpath + 'perm.txt'
}

final_output = outputs[a]

if config['starting_solution']:
    prevsol = config['starting_solution']
else:
    prevsol = outpath + "imat_solution.csv"

if config['reaction_list']:
    rlstring = '-l '+config['reaction_list']
elif a == 'div':
    rlstring = ''
else:
    m = dp.read_model(config['model'])
    dp.model_functions.get_all_reactions_from_model(m, save=True, shuffle=True, out_path=outpath)
    rlstring = '-l ' + outpath + m.id + '_reactions_shuffled.csv'

if config['full']:
    fullstring = '--full'
else:
    fullstring = ''


def get_parallel():
    return list(range(config['parallel_batches']))
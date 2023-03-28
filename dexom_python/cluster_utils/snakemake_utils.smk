import ruamel.yaml as yaml
import os
from pathlib import Path

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
    1: 'logs/a_1.txt',
    2: 'logs/a_2.txt',
    3: 'logs/a_3.txt',
    'rxn': 'logs/rxn.txt',
    'icut': 'logs/icut.txt',
    'maxdist': 'logs/max.txt',
    'div': 'logs/div.txt',
}

final_output = outputs[a]

if config['starting_solution']:
    prevsol = config['starting_solution']
else:
    prevsol = outpath + "imat_solution.csv"

if config['reaction_list']:
    rlstring = '-l '+config['reaction_list']
else:
    rlstring = ''

if config['full']:
    fullstring = '--full'
else:
    fullstring = ''


def get_parallel():
    return list(range(config['parallel_batches']))


if a == 2:
    paths = Path(outpath).glob('*solution*.csv')
    startsol_num = len(paths)
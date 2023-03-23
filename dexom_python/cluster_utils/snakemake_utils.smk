import ruamel.yaml as yaml
import pandas as pd
import os

# read configuration from YAML files
yaml_reader = yaml.YAML(typ='safe')
with open('cluster_config.yaml', 'r') as file:
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
    1: 'a_1.txt',
    2: 'a_2.txt',
    3: 'a_3.txt',
    'rxn': 'rxn.txt',
    'icut': 'icut.txt',
    'max': 'max.txt',
    'div': 'div.txt',
}

final_output = outputs[a]


def get_parallel():
    return list(range(config['parallel_batches']))

import os
import sys
import ruamel.yaml as yaml

# read configuration from YAML files
yaml_reader = yaml.YAML(typ='safe')

with open('dexom_python/cluster_utils/cluster_config.yaml', 'r') as file:
    c = file.read()
clus = yaml_reader.load(c)

cmdline = ["sbatch"]

mem_gb = clus['memory']
threads = clus['cores']
runtime = clus['time']

if clus['suppress_slurmfiles']:
    cmdline.append('--output=/dev/null --error=/dev/null')

slurm_args = f" --mem {mem_gb}G -c {threads} -t {runtime}"
cmdline.append(slurm_args)

dependencies = list(sys.argv[1:-1])
if dependencies:
    cmdline.append("--dependency")
    dependencies = [x for x in dependencies if x.isdigit()]
    cmdline.append("afterok:" + ",".join(dependencies))

jobscript = sys.argv[-1]
cmdline.append(jobscript)

os.system(" ".join(cmdline))
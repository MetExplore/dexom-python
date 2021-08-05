
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import load_json_model, load_matlab_model, read_sbml_model
from src.result_functions import read_solution
from scipy.spatial.distance import pdist, squareform
from src.model_functions import get_all_reactions_from_model


class EnumSolution(object):
    def __init__(self, solutions, binary, objective_value):
        self.solutions = solutions
        self.binary = binary
        self.objective_value = objective_value


def get_recent_solution_and_iteration(dirpath, startsol_num):
    paths = sorted(list(Path(dirpath).glob("*solution_*.csv")), key=os.path.getctime)
    paths.reverse()
    solpath = paths[int(np.random.exponential(5))]
    solution, binary = read_solution(solpath)
    iteration = len(paths) + 1 - startsol_num
    return solution, iteration


def write_rxn_enum_script(directory, modelfile, weightfile, reactionlist, imatsol, iters=100):
    with open(reactionlist, "r") as file:
        rxns = file.read().split("\n")
    rxn_num = (len(rxns) // iters) + 1
    for i in range(rxn_num):
        with open(directory+"/file_" + str(i) + ".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 05:00:00\n#SBATCH -J rxn_%i\n#SBATCH -o rxnout_%i.out\n#SBATCH -e rxnerr_%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            f.write('python src/enum_functions/rxn_enum.py -o %s/rxn_enum_%i --range %i_%i -m %s -r %s -l %s -p %s '
                    '-t 6000 --save\n' % (directory, i, i*iters, i*iters+iters, modelfile, weightfile, reactionlist,
                                          imatsol))
    with open(directory+"/runfiles.sh", "w+") as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (rxn_num-1))


def write_batch_script1(directory, username, modelfile, weightfile, reactionlist, imatsol, objtol, filenums=100, iters=100):
    for i in range(filenums):
        with open(directory+"/file_"+str(i)+".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 10:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/%s/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/%s/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n' % (username, username))
            f.write('python src/enum_functions/rxn_enum.py -o %s/rxn_enum_%i --range %i_%i -m %s -r %s -l %s -p %s '
                    '-t 600 --save\n' % (directory, i, i*5, i*5+5, modelfile, weightfile, reactionlist, imatsol))
            a = (1-1/(filenums*2*(iters/10)))**i
            f.write('python src/enum_functions/diversity_enum.py -o %s/div_enum_%i -m %s -r %s -p '
                    '%s/rxn_enum_%i_solution_0.csv -a %.5f -i %i --obj_tol %.4f'
                    % (directory, i, modelfile, weightfile, directory, i, a, iters, objtol))
    with open(directory+"/runfiles.sh", "w+") as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (filenums-1))
    return True


def write_batch_script2(filenums):
    paths = sorted(list(Path("parallel_approach2/").glob("*solution_*.csv")), key=os.path.getctime)
    paths.reverse()
    for i in range(filenums):
        with open("parallel_approach2/rxnstart_"+str(i)+".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 00:05:00\n#SBATCH -J dexom2_%i\n#SBATCH -o dex2out%i.out\n#SBATCH -e dex2err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            sol = str(paths[i]).replace("\\", "/")
            f.write('python src/enum_functions/diversity_enum.py -o parallel_approach2/div_enum_%i_0 -m '
                    'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p %s -i 1 -a 0.99 '
                    '--save --full' % (i, sol))
    with open("parallel_approach2/dexomstart.sh", "w+") as f:
        f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                '#SBATCH -t 01:00:00\n')
        f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                '/x86-64_linux"\n')
        f.write('python src/enum_functions/diversity_enum.py -o parallel_approach2/div_enum -m '
                'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p parallel_approach2 '
                '-i 1 -a 0.99 -s 100 --save --full')
    with open("parallel_approach2/rundexoms.sh", "w+") as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J rundexoms\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix rxnstart_"$i".sh\n    sbatch '
                'rxnstart_"$i".sh\ndone\ndos2unix dexomstart.sh\nfor i in {0..%i}\ndo\n    sbatch -J dexomiter_"$i" '
                '-o dexout_"$i".out -e dexerr_"$i".out dexomstart.sh \ndone' % (filenums-1, filenums-1))
    return True


def analyze_div_enum_results(result_path, solution_path, out_path):
    """
    This function calculates the average pairwise hamming distance and average next neighbour distance
    for each iteration - it's very slow

    Parameters
    ----------
    result_path: csv results file from diversity-enum
    solution_path: csv solution file from diversity-enum
    out_path: path for saving

    Returns
    -------

    """

    res = pd.read_csv(result_path, index_col=0)
    sol = pd.read_csv(solution_path, index_col=0)

    unique = len(sol.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(sol)-unique))

    time = res["time"].cumsum()
    print("Total computation time: %i s" % time.iloc[-1])
    print("Average time per iteration: %i s" % (time.iloc[-1]/len(sol)))

    fig = time.plot().get_figure()
    fig.savefig(out_path + "_cumulated_time.png")
    plt.clf()
    fig = res["selected reactions"].plot().get_figure()
    fig.savefig(out_path + "_selected_reactions.png")

    sol = sol.drop_duplicates()
    avg_pairwise = []
    avg_nearest = []

    for i in range(2, len(sol) + 1):
        distances = pdist(sol[:i].values, metric="hamming")
        avg_pairwise.append(distances.mean())
        dist_mat = squareform(distances)
        avg_nearest.append(sum([min(x[np.nonzero(x)]) for x in dist_mat])/i)

    x = range(len(avg_pairwise))

    plt.clf()
    plt.plot(x, avg_pairwise, 'r')
    plt.savefig(out_path + "_avg_pairwise.png")
    plt.clf()
    plt.plot(x, avg_nearest, 'g')
    plt.savefig(out_path + "_avg_nearest_neighbor.png")
    plt.clf()
    fig = time.plot().get_figure()
    fig.savefig(out_path + "_cumulated_time.png")
    plt.clf()
    fig = res["selected reactions"].plot().get_figure()
    fig.savefig(out_path + "_selected_reactions.png")
    return sol.T


if __name__ == "__main__":
    description = "Writes batch scripts for launching DEXOM on a slurm cluster. Note that default parameters are used."

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--directory", default="", help="directory in which to write")
    parser.add_argument("-u", "--username", help="username on the slurm cluster")
    parser.add_argument("-m", "--model", default=None, help="Metabolic model in sbml, json, or matlab format")
    parser.add_argument("-r", "--reaction_weights", default=None,
                        help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-l", "--reaction_list", default=None, help="shuffled list of reactions in the model")
    parser.add_argument("-p", "--prev_sol", help="starting solution [not optional here]")
    parser.add_argument("--obj_tol", type=float, default=1e-2,
                        help="objective value tolerance, as a fraction of the original value")
    parser.add_argument("-n", "--filenums", type=int, default=100, help="number of parallel threads")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="number of div-enum iterations per thread")

    args = parser.parse_args()

    if args.reaction_list:
        reactionlist = args.reaction_list
    else:
        fileformat = Path(args.model).suffix
        if fileformat == ".sbml" or fileformat == ".xml":
            model = read_sbml_model(args.model)
        elif fileformat == '.json':
            model = load_json_model(args.model)
        elif fileformat == ".mat":
            model = load_matlab_model(args.model)
        else:
            print("Only SBML, JSON, and Matlab formats are supported for the models")
            model = None
        get_all_reactions_from_model(model, save=True, shuffle=True, out_path=args.directory)
        reactionlist = args.directory+"/"+model.id+"_reactions_shuffled.csv"

    write_batch_script1(directory=args.directory, username=args.username, modelfile=args.model,
                        weightfile=args.reaction_weights, reactionlist=reactionlist, imatsol=args.prev_sol,
                        objtol=args.obj_tol, filenums=args.filenums, iters=args.iterations)


import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dexom_python.result_functions import read_solution
from dexom_python.imat_functions import create_new_partial_variables, create_full_variables
from scipy.spatial.distance import pdist, squareform
from dexom_python.model_functions import read_model, get_all_reactions_from_model


class EnumSolution(object):
    """
    class for solutions of enumeration methods

    Parameters
    ----------
    solutions: list
        list of pandas dataframes containing flux values with reaction ids as index
    binary: list
        list containing binary arrays of reaction activity (0 for inactive, 1 for active)
    objective_value: float
        objective value returned by the solver at the end of the optimization
    """
    def __init__(self, solutions, binary, objective_value):
        """

        Parameters
        ----------
        solutions: list
            list of pandas dataframes containing flux values with reaction ids as index
        binary: list
            list containing binary arrays of reaction activity (0 for inactive, 1 for active)
        objective_value: float
            objective value returned by the solver at the end of the optimization
        """
        self.solutions = solutions
        self.binary = binary
        self.objective_value = objective_value


def create_enum_variables(model, reaction_weights, eps=1e-2, thr=1e-5, full=False):
    for rid in reaction_weights.keys():
        if reaction_weights[rid] == 0:
            pass
        elif full and 'x_'+rid not in model.solver.variables:
            model = create_full_variables(model=model, reaction_weights=reaction_weights, epsilon=eps, threshold=thr)
            break
        elif not full and 'rh_'+rid+'_pos' not in model.solver.variables and 'rl_'+rid not in model.solver.variables:
            model = create_new_partial_variables(model=model, reaction_weights=reaction_weights, epsilon=eps,
                                                 threshold=thr)  # uses new variable implementation
            break
    return model


def get_recent_solution_and_iteration(dirpath, startsol_num):
    paths = sorted(list(Path(dirpath).glob('*solution_*.csv')), key=os.path.getctime)
    paths.reverse()
    solpath = paths[int(np.random.exponential(5))]
    solution, binary = read_solution(solpath)
    iteration = len(paths) + 1 - startsol_num
    return solution, iteration


def write_rxn_enum_script(directory, modelfile, weightfile, imatsol=None, reactionlist=None, eps=1e-4, thr=1e-5,
                          tol=1e-8, iters=100, maxiters=1e10):
    if reactionlist is not None:
        with open(reactionlist, 'r') as file:
            rxns = file.read().split('\n')
        n_max = len(rxns) if len(rxns) < maxiters else maxiters
        rxn_num = (n_max // iters) + 1
        rstring = '-l ' + reactionlist
    else:
        rstring = ''
    if imatsol is not None:
        istring = '-p ' + imatsol
    else:
        istring = ''
    for i in range(rxn_num):
        with open(directory+'/rxn_file_' + str(i) + '.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 10:00:00\n#SBATCH -J rxn_%i\n#SBATCH -o rxnout_%i.out\n#SBATCH -e rxnerr_%i.out\n'
                    % (i, i, i))
            f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/%s/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            f.write('python dexom_python/enum_functions/rxn_enum_functions.py -o %s/rxn_enum_%i --range %i_%i -m %s -r %s %s '
                    '%s -t 6000 --save -e %s --threshold %s --tol %s\n' % (directory, i, i*iters, i*iters+iters,
                    modelfile, weightfile, rstring, istring, eps, thr, tol))
    with open(directory+'/rxn_runfiles.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (rxn_num-1))


def write_batch_script_divenum(directory, modelfile, weightfile, rxnsols, objtol, eps=1e-4, thr=1e-5,
                               tol=1e-8, filenums=100, iters=100, t=6000):
    for i in range(filenums):
        with open(directory+'file_'+str(i)+'.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 05:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/%s/save/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            a = (1-1/(filenums*2*(iters/10)))**i
            f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o %sdiv_enum_%i -m %s -r %s -p '
                    '%s%s_solution_%i.csv -a %.5f -i %i --obj_tol %.4f -e %s --threshold %s --tol %s -t %i'
                    % (directory, i, modelfile, weightfile, directory, rxnsols, i, a, iters, objtol, eps, thr, tol, t))
    with open(directory+'runfiles.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (filenums-1))
    return True


def write_batch_script1(directory, modelfile, weightfile, cplexpath, reactionlist=None, imatsol=None, objtol=1e-2, filenums=100, iters=100):
    if reactionlist is not None:
        rstring = '-l ' + reactionlist
    else:
        rstring = ''
    if imatsol is not None:
        istring = '-p ' + imatsol
    else:
        istring = ''
    for i in range(filenums):
        with open(directory+'file_'+str(i)+'.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 12:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"%s'
                    '/x86-64_linux"\n' % cplexpath)
            f.write('python dexom_python/enum_functions/rxn_enum_functions.py -o %srxn_enum_%i --range %i_%i -m %s -r %s %s %s '
                    '-t 600 --save\n' % (directory, i, i*5, i*5+5, modelfile, weightfile, rstring, istring))
            a = (1-1/(filenums*2*(iters/10)))**i
            f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o %sdiv_enum_%i -m %s -r %s -p '
                    '%srxn_enum_%i_solution_1.csv -a %.5f -i %i --obj_tol %.4f'
                    % (directory, i, modelfile, weightfile, directory, i, a, iters, objtol))
    with open(directory+'runfiles.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (filenums-1))
    return True


def write_batch_script2(filenums):
    """
    Warning: this function has not been updated with most recent changes to DEXOM
    """
    paths = sorted(list(Path('parallel_approach2/').glob('*solution_*.csv')), key=os.path.getctime)
    paths.reverse()
    for i in range(filenums):
        with open('parallel_approach2/rxnstart_'+str(i)+'.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 00:05:00\n#SBATCH -J dexom2_%i\n#SBATCH -o dex2out%i.out\n#SBATCH -e dex2err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            sol = str(paths[i]).replace('\\', '/')
            f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o parallel_approach2/div_enum_%i_0 -m '
                    'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p %s -i 1 -a 0.99 '
                    '--save --full' % (i, sol))
    with open('parallel_approach2/dexomstart.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                '#SBATCH -t 01:00:00\n')
        f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                '/x86-64_linux"\n')
        f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o parallel_approach2/div_enum -m '
                'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p parallel_approach2 '
                '-i 1 -a 0.99 -s 100 --save --full')
    with open('parallel_approach2/rundexoms.sh', 'w+') as f:
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
    print('There are %i unique solutions and %i duplicates' % (unique, len(sol)-unique))
    time = res['time'].cumsum()
    print('Total computation time: %i s' % time.iloc[-1])
    print('Average time per iteration: %i s' % (time.iloc[-1]/len(sol)))
    fig = time.plot().get_figure()
    fig.savefig(out_path + '_cumulated_time.png')
    plt.clf()
    fig = res['selected reactions'].plot().get_figure()
    fig.savefig(out_path + '_selected_reactions.png')
    sol = sol.drop_duplicates()
    avg_pairwise = []
    avg_nearest = []
    for i in range(2, len(sol) + 1):
        distances = pdist(sol[:i].values, metric='hamming')
        avg_pairwise.append(distances.mean())
        dist_mat = squareform(distances)
        avg_nearest.append(sum([min(x[np.nonzero(x)]) for x in dist_mat])/i)
    x = range(len(avg_pairwise))
    plt.clf()
    plt.plot(x, avg_pairwise, 'r')
    plt.savefig(out_path + '_avg_pairwise.png')
    plt.clf()
    plt.plot(x, avg_nearest, 'g')
    plt.savefig(out_path + '_avg_nearest_neighbor.png')
    plt.clf()
    fig = time.plot().get_figure()
    fig.savefig(out_path + '_cumulated_time.png')
    plt.clf()
    fig = res['selected reactions'].plot().get_figure()
    fig.savefig(out_path + '_selected_reactions.png')
    return sol.T


if __name__ == '__main__':
    description = 'Writes batch scripts for launching DEXOM on a slurm cluster. Note that default parameters are used.'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_path', default='cluster/', help='Path to which the files are written. '
                                                                     'Has to be a folder in this directory')
    parser.add_argument('-m', '--model', default=None, help='Metabolic model in sbml, json, or matlab format')
    parser.add_argument('-r', '--reaction_weights', default=None,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-l', '--reaction_list', default=None, help='list of reactions in the model')
    parser.add_argument('-p', '--prev_sol', default=None, help='starting solution')
    parser.add_argument('-c', '--cplex_path', default='/home/mstingl/save/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux',
                        help='path to the cplex solver')
    parser.add_argument('--obj_tol', type=float, default=1e-2,
                        help='objective value tolerance, as a fraction of the original value')
    parser.add_argument('-n', '--filenums', type=int, default=100, help='number of parallel threads')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='number of div-enum iterations per thread')

    args = parser.parse_args()

    if args.reactionlist:
        rxnlist = args.reactionlist
    else:
        model = read_model(args.model)
        get_all_reactions_from_model(model, save=True, shuffle=True, out_path=args.out_path)
        reactionlist = args.out_path + model.id + '_reactions_shuffled.csv'

    write_batch_script1(args.out_path, args.model, args.reaction_weights, args.cplex_path, reactionlist,
                        args.prev_sol, args.obj_tol, args.filenums, args.iterations)

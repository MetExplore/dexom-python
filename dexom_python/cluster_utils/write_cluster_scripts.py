import os
import argparse
from pathlib import Path
from dexom_python.model_functions import read_model, get_all_reactions_from_model, DEFAULT_VALUES


def write_rxn_enum_script(directory, modelfile, weightfile, cplexpath, imatsol=None, reactionlist=None,
                          objtol=DEFAULT_VALUES['obj_tol'], eps=DEFAULT_VALUES['epsilon'],
                          thr=DEFAULT_VALUES['threshold'], tol=DEFAULT_VALUES['tolerance'], iters=100, maxiters=1e10):
    os.makedirs(directory, exist_ok=True)
    if directory[-1] not in ['/', '\\']:
        directory += '/'
    if reactionlist is not None:
        with open(reactionlist, 'r') as file:
            rxns = file.read().split('\n')
        n_max = len(rxns) if len(rxns) < maxiters else maxiters
        rxn_num = (n_max // iters) + 1
        rstring = '-l ' + reactionlist
    else:
        rstring = ''
        model = read_model(modelfile)
        rxn_num = len(model.reactions)
    if imatsol is not None:
        istring = '-p ' + imatsol
    else:
        istring = ''
    for i in range(rxn_num):
        with open(directory+'rxn_file_' + str(i) + '.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 10:00:00\n#SBATCH -J rxn_%i\n#SBATCH -o rxnout_%i.out\n#SBATCH -e rxnerr_%i.out\n'
                    % (i, i, i))
            f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"%s"\n' % cplexpath)
            f.write('python dexom_python/enum_functions/rxn_enum_functions.py -o %srxn_enum_%i --range %i_%i -m %s -r '
                    '%s %s %s -t 6000 -e %s --threshold %s --tol %s --obj_tol %s\n'
                    % (directory, i, i*iters, i*iters+iters, modelfile, weightfile, rstring, istring, eps, thr, tol,
                       objtol))
    with open(directory+'rxn_runfiles.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (rxn_num-1))
    with open(directory+'compile_solutions.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J compile\n#SBATCH -o compout.out\n#SBATCH '
                '-e comperr.out\ncd $SLURM_SUBMIT_DIR\n'
                'python dexom_python/cluster_utils/solution_compilation.py -p "*solutions.csv" -i %s -o %s' %
                (directory, directory))


def write_batch_script_divenum(directory, modelfile, weightfile, cplexpath, rxnsols, objtol, filenums=100, iters=100,
                               eps=DEFAULT_VALUES['epsilon'], thr=DEFAULT_VALUES['threshold'],
                               tol=DEFAULT_VALUES['tolerance'], t=DEFAULT_VALUES['timelimit']):
    os.makedirs(directory, exist_ok=True)
    if directory[-1] not in ['/', '\\']:
        directory += '/'
    for i in range(filenums):
        with open(directory+'file_'+str(i)+'.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 05:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"%s"\n' % cplexpath)
            a = (1-1/(filenums*2*(iters/10)))**i
            f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o %sdiv_enum_%i -m %s -r %s -p '
                    '%s%s_solution_%i.csv -a %.5f -i %i --obj_tol %.4f -e %s --threshold %s --tol %s -t %i'
                    % (directory, i, modelfile, weightfile, directory, rxnsols, i, a, iters, objtol, eps, thr, tol, t))
    with open(directory+'runfiles.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (filenums-1))
    return True


def write_batch_script1(directory, modelfile, weightfile, cplexpath, reactionlist=None, imatsol=None,
                        objtol=DEFAULT_VALUES['obj_tol'], filenums=100, iters=100, rxniters=5):
    """
    Writes bash scripts for dexom-python parallelization approach 1 on a slurm cluster. Within each batch,
    reaction-enumeration and diversity-enumeration are performed. These scripts assume that you have setup
    a virtual environment called env.

    Parameters
    ----------
    directory: str
        directory in which the files will be generated. If it does not exist, it will be created
    modelfile: str
        path to the model
    weightfile:
        path to the reaction weights
    cplexpath: str
        path to a cplex installation on the cluster
    reactionlist: str
        list of reactions for reaction-enumeration
    imatsol: str
        path to imat solution
    objtol: float
        objective tolerance
    filenums: int
        number of parallel batches
    iters: int
        number of diversity-enumeration iterations per batch
    """
    os.makedirs(directory, exist_ok=True)
    if directory[-1] not in ['/', '\\']:
        directory += '/'
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
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"%s/x86-64_linux"\n' % cplexpath)
            f.write('python dexom_python/enum_functions/rxn_enum_functions.py -o %srxn_enum_%i --range %i_%i -m %s -r '
                    '%s %s %s -t 600 --save\n' % (directory, i, i*rxniters, i*rxniters+rxniters, modelfile, weightfile,
                                                  rstring, istring))
            a = (1-1/(filenums*2*(iters/10)))**i
            f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o %sdiv_enum_%i -m %s -r %s -p '
                    '%srxn_enum_%i_solution_1.csv -a %.5f -i %i --obj_tol %.4f'
                    % (directory, i, modelfile, weightfile, directory, i, a, iters, objtol))
    with open(directory+'runfiles.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (filenums-1))
    return True


def write_batch_script2(directory, modelfile, weightfile, cplexpath, objtol=DEFAULT_VALUES['obj_tol'], filenums=100):
    """
    Writes bash scripts for dexom-python parallelization approach 2 on a slurm cluster. In this approach, indiviual
    diversity-enumeration iterations are laucnhed in each batch - this requires the existance of reaction-enumeration
    solutions beforehand. These scripts assume that you have setup a virtual environment called env.

    Parameters
    ----------
    directory: str
        directory in which the files will be generated
    modelfile: str
        path to the model
    weightfile:
        path to the reaction weights
    cplexpath: str
        path to a cplex installation on the cluster
    objtol: float
        objective tolerance
    filenums: int
        number of parallel batches
    """
    os.makedirs(directory, exist_ok=True)
    if directory[-1] not in ['/', '\\']:
        directory += '/'
    paths = sorted(list(Path(directory).glob('*solution_*.csv')), key=os.path.getctime)
    paths.reverse()
    for i in range(filenums):
        a = (1 - 1 / (filenums * 2 * (filenums / 10))) ** i
        with open(directory+'rxnstart_'+str(i)+'.sh', 'w+') as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 00:05:00\n#SBATCH -J dexom2_%i\n#SBATCH -o dex2out%i.out\n#SBATCH -e dex2err%i.out\n'
                    % (i, i, i))
            f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"%s/x86-64_linux"\n' % cplexpath)
            sol = str(paths[i]).replace('\\', '/')
            f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o %sdiv_enum_%i -m %s -r %s -p '
                    '%s -a %.5f -i 1 --obj_tol %.4f --save'
                    % (directory, i, modelfile, weightfile, sol, a, objtol))
    a = (1 - 1 / (filenums * 2 * (filenums / 10)))
    with open('parallel_approach2/dexomstart.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                '#SBATCH -t 01:00:00\n')
        f.write('cd $SLURM_SUBMIT_DIR\ncd ..\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                'activate\nexport PYTHONPATH=${PYTHONPATH}:"%s/x86-64_linux"\n' % cplexpath)
        f.write('python dexom_python/enum_functions/diversity_enum_functions.py -o %sdiv_enum -m %s -r %s -p '
                '%s -a %.5f -i 1 -s %i --obj_tol %.4f --save'
                % (directory, modelfile, weightfile, sol, a, filenums, objtol))
    with open(directory+'rundexoms.sh', 'w+') as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J rundexoms\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix rxnstart_"$i".sh\n    sbatch '
                'rxnstart_"$i".sh\ndone\ndos2unix dexomstart.sh\nfor i in {0..%i}\ndo\n    sbatch -J dexomiter_"$i" '
                '-o dexout_"$i".out -e dexerr_"$i".out dexomstart.sh \ndone' % (filenums-1, filenums-1))
    return True


def main():
    """
    This function is called when you run this script from the commandline.
    It writes batch scripts for launching DEXOM on a slurm cluster.
    Note that default values are used for most parameters.
    This also assumes that you have a virtual environment called env in your project directory
    Use --help to see commandline parameters

    There are 3 approaches for using parallel batches in DEXOM:
    Approach 1: Within each batch, reaction-enumeration and diversity-enumeration are performed.
    Approach 2: Indiviual diversity-enumeration iterations are launched in each batch - this requires the existance
    of reaction-enumeration solutions beforehand.
    Approach 3: First, launch parallel reaction-enumeration batches. Then compile the solutions.
    Then diversity-enumeration batches can be launched using the compiled rxn-enum solutions as starting points.
    """
    description = 'Writes batch scripts for launching DEXOM on a slurm cluster. Note that default values are used' \
                  'for most parameters. This also assumes that you have a virtual environment called env in your' \
                  'project directory'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_path', default='cluster/', help='Folder to which the files are written. '
                                                                     'The folder will be created if not present')
    parser.add_argument('-m', '--model', default=None, help='Metabolic model in sbml, json, or matlab format')
    parser.add_argument('-r', '--reaction_weights', default=None,
                        help='Reaction weights in csv format (first row: reaction names, second row: weights)')
    parser.add_argument('-l', '--reaction_list', default=None, help='list of reactions in the model')
    parser.add_argument('-p', '--prev_sol', default=None, help='starting solution')
    parser.add_argument('-c', '--cplex_path', help='path to the cplex solver',
                        default='/home/mstingl/save/CPLEX_Studio1210/cplex/python/3.7/x86-64_linux')
    parser.add_argument('--obj_tol', type=float, default=DEFAULT_VALUES['obj_tol'],
                        help='objective value tolerance, as a fraction of the original value')
    parser.add_argument('-n', '--filenums', type=int, default=100, help='number of parallel threads')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='number of div-enum iterations per thread')
    parser.add_argument('--rxniters', type=int, default=5, help='number of rxn-enum iterations per thread (approach 1)')
    parser.add_argument('-a', '--approach', type=int, default=1, help='which parallelisation approach to use')

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    if args.reactionlist:
        reactionlist = args.reactionlist
    else:
        model = read_model(args.model)
        get_all_reactions_from_model(model, save=True, shuffle=True, out_path=args.out_path)
        reactionlist = args.out_path + model.id + '_reactions_shuffled.csv'

    if args.approach == 1:
        print('Approach 1: Within each batch, reaction-enumeration and diversity-enumeration are performed.')
        write_batch_script1(args.out_path, args.model, args.reaction_weights, args.cplex_path, reactionlist,
                            args.prev_sol, args.obj_tol, args.filenums, args.iterations, args.rxniters)
    elif args.approach == 2:
        print('Approach 2: Indiviual diversity-enumeration iterations are launched in each batch - this requires the '
              'existance of reaction-enumeration solutions beforehand.')
        write_batch_script2(args.out_path, args.model, args.reaction_weights, args.cplex_path, args.obj_tol,
                            args.filenums)
    elif args.approach == 3:
        print('Approach 3: First, launch parallel reaction-enumeration batches. Then compile the solutions. Then '
              'diversity-enumeration batches can be launched using the compiled rxn-enum solutions as starting points.')
        write_rxn_enum_script(args.out_path, args.model, args.reaction_weights, args.cplex_path, args.prev_sol,
                              reactionlist, args.obj_tol, DEFAULT_VALUES['epsilon'], DEFAULT_VALUES['threshold'],
                              DEFAULT_VALUES['tolerance'], args.filenums, maxiters=1e10)
        write_batch_script_divenum(args.out_path, args.model, args.reaction_weights, args.cplex_path, args.out_path,
                                   args.filenums, args.iterations, DEFAULT_VALUES['epsilon'],
                                   DEFAULT_VALUES['threshold'], DEFAULT_VALUES['tolerance'],
                                   DEFAULT_VALUES['timelimit'])
    else:
        print('approach parameter value must be 1, 2, or 3')
    return True


if __name__ == '__main__':
    main()


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model_functions import load_reaction_weights
from src.result_functions import read_solution


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


def write_rxn_enum_script(model, out_folder="recon_rxn_enum", iters=100):
    rxn_num = (len(model.reactions) // iters) + 1
    for i in range(rxn_num):
        with open(out_folder+"/file_" + str(i) + ".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 05:00:00\n#SBATCH -J rxn_%i\n#SBATCH -o rxnout_%i.out\n#SBATCH -e rxnerr_%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            f.write('python src/enum_functions/rxn_enum.py -o %s/rxn_enum_%i --range %i_%i -t 6000 '
                    '-m recon2_2/recon2v2_corrected.json -r recon2_2/pval_0-01_reactionweights.csv '
                    '-l recon2_2/recon2v2_reactions_shuffled.csv -p recon2_2/pval_0-01_imatsol.csv\n'
                    % (out_folder, i, i*iters, i*iters+iters))
    with open(out_folder+"/runfiles.sh", "w+") as f:
        f.write('#!/bin/bash\n#SBATCH --mail-type=ALL\n#SBATCH -J runfiles\n#SBATCH -o runout.out\n#SBATCH '
                '-e runerr.out\ncd $SLURM_SUBMIT_DIR\nfor i in {0..%i}\ndo\n    dos2unix file_"$i".sh\n    sbatch'
                ' file_"$i".sh\ndone' % (rxn_num-1))


def write_batch_script1(filenums, iters=10):
    for i in range(filenums):
        with open("par_1/file_"+str(i)+".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=8G\n#SBATCH -c 24\n'
                    '#SBATCH -t 05:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            f.write('python src/enum_functions/rxn_enum.py -o par_1/rxn_enum_%i --range %i_%i '
                    '-m recon2_2/recon2v2_corrected.json -r recon2_2/pval_0-01_reactionweights.csv '
                    '-l recon2_2/recon2v2_reactions_shuffled.csv -p recon2_2/pval_0-01_imatsol.csv -t 6000 '
                    '--save\n' % (i, i*5, i*5+5))
            a = (1-1/(filenums*2*(iters/10)))**i
            f.write('python src/enum_functions/diversity_enum.py -o par_1/div_enum_%i -m '
                    'recon2_2/recon2v2_corrected.json -r recon2_2/pval_0-01_reactionweights.csv -p '
                    'par_1/rxn_enum_%i_solution_0.csv -a %.5f -i %i --obj_tol 0.01' % (i, i, a, iters))
    with open("par_1/runfiles.sh", "w+") as f:
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
                    '#SBATCH -t 01:00:00\n#SBATCH -J dexom2_%i\n#SBATCH -o dex2out%i.out\n#SBATCH -e dex2err%i.out\n'
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


def dexom_results(result_path, solution_path, out_path):

    res = pd.read_csv(result_path, index_col=0)
    sol = pd.read_csv(solution_path, index_col=0)

    unique = len(sol.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(sol)-unique))

    time = res["time"].cumsum()
    print("Total computation time: %i s" % time.iloc[-1])
    print("Average time per iteration: %i s" % (time.iloc[-1]/len(sol)))

    sol = sol.T
    avg_pairwise = []
    avg_near = []
    hammings = []
    h = 0
    for x in sol:
        hammings.append([])
        if x > 0:
            for y in range(x):
                temp = sum(abs(sol[x]-sol[y]))
                h += temp
                hammings[x].append(temp)
                hammings[y].append(temp)
            avg_pairwise.append((h/(x*(x+1)/2))/len(sol))
            temp = 0
            for v in hammings:
                temp += min(v)/len(sol)
            avg_near.append(temp/x)
    x = range(len(avg_pairwise))

    plt.clf()
    plt.plot(x, avg_pairwise, 'r')
    plt.savefig(out_path + "_avg_pairwise.png")
    plt.clf()
    plt.plot(x, avg_near, 'g')
    plt.savefig(out_path + "_avg_nearest_neighbor.png")
    plt.clf()
    fig = time.plot().get_figure()
    fig.savefig(out_path + "_cumulated_time.png")
    plt.clf()
    fig = res["selected reactions"].plot().get_figure()
    fig.savefig(out_path + "_selected_reactions.png")
    return sol.T


def dexom_cluster_results(in_folder, out_folder, approach, filenums=100):

    # concatenating all .out files from the cluster
    if approach == 1:
        fileout = '/dex1out'
        fileerr = '/dex1err'
        with open(out_folder+'/all_outs.txt', 'w+') as outfile:
            for i in range(filenums):
                fname = in_folder+fileout+str(i)+'.out'
                with open(fname) as infile:
                    outfile.write(infile.read())
        with open(out_folder+'/all_errs.txt', 'w+') as outfile:
            for i in range(filenums):
                fname = in_folder+fileerr+str(i)+'.out'
                with open(fname) as infile:
                    outfile.write(infile.read())
    elif approach == 2:
        outfiles = Path(in_folder).glob("*out*.out")
        errfiles = Path(in_folder).glob("*err*.out")
        with open(out_folder + '/all_outs.txt', 'w+') as outfile:
            for f in outfiles:
                with open(str(f)) as infile:
                    outfile.write(infile.read())
        with open(out_folder + '/all_errs.txt', 'w+') as outfile:
            for f in errfiles:
                with open(str(f)) as infile:
                    outfile.write(infile.read())

    #concatenating & analyzing rxn_enum results
    print("looking at rxn_enum")

    all_rxn = []
    for i in range(filenums):
        try:
            if approach == 1:
                filename = in_folder + '/rxn_enum_%i_solutions.csv' % i
            elif approach == 2:
                filename = Path(in_folder).glob("div_enum_%i_*_solutions.csv" % i)
                filename = str(list(filename)[0])
            rxn = pd.read_csv(filename, index_col=0)
            all_rxn.append(rxn)
        except:
            pass
    rxn = pd.concat(all_rxn, ignore_index=True)
    rxn.to_csv(out_folder+"/all_rxn_sol.csv")
    if approach == 1:
        unique = len(rxn.drop_duplicates())
        print("There are %i unique solutions and %i duplicates" % (unique, len(rxn) - unique))
        fulltime = 0
        counter = 0
        with open(out_folder+"/all_outs.txt", "r") as file:
            for line in file:
                line = line.split()
                try:
                    fulltime += float(line[0])
                    counter += 1
                except:
                    pass
        if counter != 0:
            print("Total computation time:", int(fulltime), "s")
            print("Average time per iteration:", int(fulltime*2/counter), "s")
    if approach == 2:
        all_res = []
        for i in range(filenums):
            try:
                filename = Path(in_folder).glob("div_enum_%i_*_results.csv" % i)
                filename = str(list(filename)[0])
                res = pd.read_csv(filename, index_col=0)
                all_res.append(res)
            except:
                pass
        rxn_res = pd.concat(all_res, ignore_index=True)
        rxn_res.to_csv(out_folder + "/all_rxn_res.csv")
        dexom_results(out_folder + "/all_rxn_res.csv", out_folder + "/all_rxn_sol.csv", out_folder + "/all_rxn_dexom")

    # concatenating & analyzing diversity_enum results
    print("looking at diversity_enum")
    all_res = []
    all_sol = []
    if approach == 1:
        for i in range(filenums):
            try:
                solname = in_folder + '/div_enum_%i_solutions.csv' % i
                resname = in_folder + '/div_enum_%i_results.csv' % i
                sol = pd.read_csv(solname, index_col=0)
                res = pd.read_csv(resname, index_col=0)
                all_sol.append(sol)
                all_res.append(res)
            except:
                pass
    elif approach == 2:
        solname = Path(in_folder).glob("div_enum2021*_solutions.csv")
        all_sol = [pd.read_csv(str(x), index_col=0) for x in solname]
        resname = Path(in_folder).glob("div_enum2021*_results.csv")
        all_res = [pd.read_csv(str(x), index_col=0) for x in resname]
    sol = pd.concat(all_sol, ignore_index=True)
    res = pd.concat(all_res, ignore_index=True)
    sol.to_csv(out_folder+"/all_sol.csv")
    res.to_csv(out_folder+"/all_res.csv")
    unique = len(sol.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(sol)-unique))

    time = res["time"].cumsum()
    print("Total computation time: %i s" % time.iloc[-1])
    print("Average time per iteration: %i s" % (time.iloc[-1]/len(sol)))

    plt.clf()
    fig = res.sort_values("selected reactions").reset_index(drop=True)["selected reactions"].plot().get_figure()
    fig.savefig(out_folder+"/all_dexom_selected_reactions_ordered.png")

    # analyzing total results
    print("total result")
    full = pd.concat([rxn, sol], ignore_index=True)
    unique = len(full.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(full)-unique))

    return full.drop_duplicates()


if __name__ == "__main__":
    from cobra.io import load_json_model

    model = load_json_model("recon2_2/recon2v2_corrected.json")
    write_rxn_enum_script(model, iters=100)
    #
    # sol = dexom_cluster_results("par_1_obj001", "par_1_obj001_an", approach=1, filenums=100)

    # folder = "par_1_newobjtol_an/"
    # ol = dexom_results(folder+"all_res.csv", folder+"all_sol.csv", folder+"obj")

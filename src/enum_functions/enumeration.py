
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.imat import imat
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


def write_batch_script1(filenums):
    for i in range(filenums):
        with open("parallel_approach1/file_"+str(i)+".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 02:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            f.write('python src/enum_functions/rxn_enum.py -o parallel_approach1/rxn_enum_%i --range %i_%i '
                    '-m min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv '
                    '-l min_iMM1865/min_iMM1865_reactions_shuffled.txt -t 600\n' % (i, i*5, i*5+10))
            a = (1-1/(filenums*2))**i
            f.write('python src/enum_functions/diversity_enum.py -o parallel_approach1/div_enum_%i -m '
                    'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p '
                    'parallel_approach1/rxn_enum_%i_solution_0.csv -a %.5f' % (i, i, a))
    return True


def write_batch_script2(filenums):
    paths = sorted(list(Path("parallel_approach2").glob("*solution_*.csv")), key=os.path.getctime)
    paths.reverse()
    for i in range(filenums):
        with open("parallel_approach2/rxnstart_"+str(i)+".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 00:05:00\n#SBATCH -J dexom2_%i\n#SBATCH -o dex2out%i.out\n#SBATCH -e dex2err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            sol = paths[i]
            f.write('python src/enum_functions/diversity_enum.py -o parallel_approach2/div_enum_%i_0 -m '
                    'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p %s -i 1 -a 0.995 '
                    '--save' % (i, sol))
    with open("parallel_approach2/dexomstart.sh", "w+") as f:
        f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                '#SBATCH -t 00:10:00\n')
        f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                '/x86-64_linux"\n')
        f.write('python src/enum_functions/diversity_enum.py -o parallel_approach2/div_enum -m '
                'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p parallel_approach2 '
                '-i 1 -a 0.995 --save')


def dexom_results(result_path, solution_path, out_path):

    res = pd.read_csv(result_path, index_col=0)
    df = pd.read_csv(solution_path, index_col=0)

    unique = len(df.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(df)-unique))

    time = res["time"].cumsum()
    print("Total computation time: %i s" % time.iloc[-1])
    print("Average time per iteration: %i s" % (time.iloc[-1]/len(df)))

    df = df.T
    avg_pairwise = []
    avg_near = []
    hammings = []
    h = 0
    for x in df:
        hammings.append([])
        if x > 0:
            for y in range(x):
                temp = sum(abs(df[x]-df[y]))
                h += temp
                hammings[x].append(temp)
                hammings[y].append(temp)
            avg_pairwise.append((h/(x*(x+1)/2))/len(df))
            temp = 0
            for v in hammings:
                temp += min(v)/len(df)
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
    return df.T


def dexom_cluster_results(in_folder, out_folder):

    # concatenating all .out files from the cluster
    with open(out_folder+'/all_outs.txt', 'w+') as outfile:
        for i in range(100):
            fname = in_folder+'/rxnout'+str(i)+'.out'
            with open(fname) as infile:
                outfile.write(infile.read())
    with open(out_folder+'/all_errs.txt', 'w+') as outfile:
        for i in range(100):
            fname = in_folder+'/rxnerr'+str(i)+'.out'
            with open(fname) as infile:
                outfile.write(infile.read())

    #concatenating & analyzing rxn_enum results
    print("looking at rxn_enum")
    all_rxn = []
    for i in range(100):
        try:
            rxn = pd.read_csv(in_folder+'/rxn_enum_%i_solutions.csv' % i, index_col=0)
            all_rxn.append(rxn)
        except:
            pass
    rxn = pd.concat(all_rxn, ignore_index=True)
    rxn.to_csv(out_folder+"/all_rxn.csv")
    unique = len(rxn.drop_duplicates())
    print("There are %i unique solutions and %i duplicates" % (unique, len(rxn)-unique))
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
    print("Total computation time:", fulltime)
    print("Average time per iteration:", fulltime*2/counter)

    # concatenating & analyzing diversity_enum results
    print("looking at diversity_enum")
    all_res = []
    all_sol = []
    for i in range(100):
        try:
            sol = pd.read_csv(in_folder+'/div_enum_%i_solutions.csv' % i, index_col=0)
            res = pd.read_csv(in_folder+'/div_enum_%i_results.csv' % i, index_col=0)
            all_sol.append(sol)
            all_res.append(res)
        except:
            pass
    sol = pd.concat(all_sol, ignore_index=True)
    res = pd.concat(all_res, ignore_index=True)
    sol.to_csv(out_folder+"/all_sol.csv")
    res.to_csv(out_folder+"/all_res.csv")
    dexom_results(out_folder+"/all_res.csv", out_folder+"/all_sol.csv", out_folder+"/all_dexom")

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

    write_batch_script2(100)

    # sol = dexom_cluster_results("parallel_approach1", "parallel_approach1_analysis")
    #
    # # calculating objective values
    # recs = load_reaction_weights("min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv")
    # imsol = read_solution("parallel_approach1/rxn_enum_0_solution_0.csv")
    # weights = np.array([recs[key] if key in recs else 0 for key in imsol[0].fluxes.index])
    # obj = [np.dot(np.array(s[1]), weights) for s in sol.iterrows()]
    # print("objective value range: ", min(obj), ",", max(obj))

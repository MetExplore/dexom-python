
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EnumSolution(object):
    def __init__(self, solutions, binary, objective_value):
        self.solutions = solutions
        self.binary = binary
        self.objective_value = objective_value


def write_batch_script(filenums):
    for i in range(filenums):
        with open("BATCH/file_"+str(i)+".sh", "w+") as file:
            file.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                       '#SBATCH -t 00:10:00\n#SBATCH -J rxn_enum_%i\n#SBATCH -o rxnout%i.out\n#SBATCH -e rxnerr%i.out\n'
                       % (i, i, i))
            file.write('cd $SLURM_SUBMIT_DIR\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/activate\n'
                       'export PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7/'
                       'x86-64_linux"\npython src/rxn_enum.py -o parallel/rxn_enum_%i --range %i0_%i0 -m '
                       'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -l '
                       'min_iMM1865/min_iMM1865_reactions.txt -t 600' % (i, i, i+1))
    return 0


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


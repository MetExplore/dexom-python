
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


def write_batch_script1(filenums):
    for i in range(filenums):
        with open("parallel_approach1/file_"+str(i)+".sh", "w+") as f:
            f.write('#!/bin/bash\n#SBATCH -p workq\n#SBATCH --mail-type=ALL\n#SBATCH --mem=64G\n#SBATCH -c 24\n'
                    '#SBATCH -t 01:00:00\n#SBATCH -J dexom1_%i\n#SBATCH -o dex1out%i.out\n#SBATCH -e dex1err%i.out\n'
                    % (i, i, i))
            f.write('cd /home/mstingl/work/dexom_py\nmodule purge\nmodule load system/Python-3.7.4\nsource env/bin/'
                    'activate\nexport PYTHONPATH=${PYTHONPATH}:"/home/mstingl/work/CPLEX_Studio1210/cplex/python/3.7'
                    '/x86-64_linux"\n')
            # f.write('python src/enum_functions/rxn_enum.py -o parallel_approach1/rxn_enum_%i --range %i_%i '
            #         '-m min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv '
            #         '-l min_iMM1865/min_iMM1865_reactions_shuffled.txt -t 6000\n' % (i, i*5, i*5+10))
            a = (1-1/(filenums*2))**i
            f.write('python src/enum_functions/diversity_enum.py -o parallel_approach1/div_enum_%i -m '
                    'min_iMM1865/min_iMM1865.xml -r min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv -p '
                    'parallel_approach1/rxn_enum_%i_solution_1.csv -a %.5f' % (i, i, a))
    with open("parallel_approach1/runfiles.sh", "w+") as f:
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
    from cobra.io import read_sbml_model
    # write_batch_script1(100)

    # sol = dexom_cluster_results("parallel_approach1", "parallel_approach1_analysis", approach=1, filenums=100)

    sol4 = pd.read_csv("4_all_sol.csv", index_col=0)
    sol5 = pd.read_csv("5_all_sol.csv", index_col=0)
    doubsol = pd.concat([sol4, sol5], ignore_index=True)
    sol = doubsol.drop_duplicates()

    # calculating objective values
    recs = load_reaction_weights("min_iMM1865/p53_deseq2_cutoff_padj_1e-6.csv")
    model = read_sbml_model("min_iMM1865/min_iMM1865.xml")
    weights = np.array([recs.get(rxn.id, 0) for rxn in model.reactions])
    obj = [np.dot(np.array(s[1]), weights) for s in sol.iterrows()]
    print("objective value range: ", min(obj), ",", max(obj))

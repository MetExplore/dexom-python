
from imat import imat
from result_functions import write_solution


def rxn_enum_single_loop(model, reaction_weights, rec_id, new_rec_state, out_name, eps=1e-2, thr=1e-5, tlim=None, feas=1e-6, mipgap=1e-3):
    with model as model_temp:
        if rec_id not in model.reactions:
            print("reaction not found in model")
            return 0
        rxn = model_temp.reactions.get_by_id(rec_id)
        if int(new_rec_state) == 0:
            rxn.bounds = (0., 0.)
        elif int(new_rec_state) == 1:
            rxn.lower_bound = thr
        elif int(new_rec_state) == 2:
            rxn.upper_bound = -thr
        else:
            print("new_rec_state has an incorrect value: %s" % str(new_rec_state))
            return 0
        try:
            sol = imat(model_temp, reaction_weights, epsilon=eps, threshold=thr, timelimit=tlim,
                            feasibility=feas, mipgaptol=mipgap)
        except:
            print("This constraint renders the problem unfeasible")
            return 0
    write_solution(sol, thr, out_name)
    return 1


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


if __name__ == "__main__":
    write_batch_script(100)
    pass

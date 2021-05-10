
from imat import imat
from result_functions import write_solution


def rxn_enum_single_loop(model, reaction_weights, rec_id, new_rec_state, out_name, eps, thr, tlim, feas, mipgap):
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

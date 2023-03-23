include: "snakemake_utils.smk"

rule end:
    input:
        final_output

rule concat_div_sols:
    input:
        expand(cluspath+"div_enum_{parallel}_solutions.csv", parallel=get_parallel())
    output:
        outpath+"all_DEXOM_solutions.csv",
        outpath+"activation_frequency_reactions.csv"
    log:
        "logs/solution_compilation.log"
    shell:
        "python cluster_utils/solution_compilation.py"

rule div_enum:
    input:
        outpath+"reaction_weights_{condition}.csv",
        cluspath+"full_rxn_enum_solutions_{condition}.csv"
    output:
        cluspath+"div_enum_stats_{condition}_{parallel}.csv",
        cluspath+"div_enum_solutions_{condition}_{parallel}.csv"
    params:
        dist_anneal = lambda w: (1 - 1 / (clus['batch_num'] * 2 * (clus['batch_div_sols'] / 10))) ** int(w.parallel)
    log:
        "logs/rxn_enum_{condition}_{parallel}.log"
    shell:
        "python utilities_cluster/cluster_div_enum.py -c {wildcards.condition} -p {wildcards.parallel} -d {params.dist_anneal} -i "+str(clus['batch_div_sols'])

rule concat_rxn_sols:
    input:
        expand(cluspath+"rxn_enum_solutions_{condition}_{parallel}.csv", condition=get_conditions(), parallel=get_parallel())
    output:
        expand(cluspath+"full_rxn_enum_solutions_{condition}.csv", condition=get_conditions())
    log:
        "logs/concat_rxn_enum.log"
    shell:
        "python utilities_cluster/cluster_concat_rxn_solutions.py"

rule rxn_enum:
    input:
        outpath+"reaction_weights_{condition}.csv",
        outpath+"imat_solution_{condition}.csv"
    output:
        cluspath+"rxn_enum_solutions_{condition}_{parallel}.csv"
    params:
        rxn_range = lambda w: str(clus['batch_rxn_sols']*int(w.parallel)) + '_' + str(clus['batch_rxn_sols']*(int(w.parallel)+1))
    log:
        "logs/rxn_enum_{condition}_{parallel}.log"
    shell:
        "python utilities_cluster/cluster_rxn_enum.py -c {wildcards.condition} -p {wildcards.parallel} -r {params.rxn_range}"

rule imat:
    input:
        doc['modelpath'],
        doc['expressionfile']
    output:
        outpath+"reaction_weights_{condition}.csv",
        outpath+"imat_solution_{condition}.csv"
    log:
        "logs/weights_imat_{condition}.log"
    shell:
        "python utilities_cluster/cluster_weights_imat.py -c {wildcards.condition}"

rule approach_1:

rule approach_2

rule approach_3:

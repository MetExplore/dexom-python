include: "snakemake_utils.smk"

rule end:
    input:
        final_output

rule concat_solutions:
    input:
        expand(cluspath+"div_enum_{parallel}_solutions.csv", parallel=get_parallel())
    output:
        outpath+"all_DEXOM_solutions.csv",
        outpath+"activation_frequency_reactions.csv"
    log:
        "logs/solution_compilation.log"
    shell:
        "python cluster_utils/solution_compilation.py -" # DEFINE the match pattern

rule div_enum:
    input:
        config['model'],
        config['reaction_weights']
    output:
        outpath+"div_enum_{parallel}_solutions.csv"
    params:
        dist_anneal = lambda w: (1 - 1 / (clus['batch_num'] * 2 * (clus['batch_div_sols'] / 10))) ** int(w.parallel)
    log:
        "logs/rxn_enum_{condition}_{parallel}.log"
    shell:
        "python dexom_python/enum_functions/diversity_enum_functions.py -m %s -r %s -a {params.dist_anneal} -o div_enum_{wildcards.parallel} %s" %
        (config['model'], config['reaction-weights'], prevstring)


rule rxn_enum:
    input:
        config['model'],
        config['reaction_weights']
    output:
        outpath+"rxn_enum_{parallel}_solutions.csv"
    params:
        rxn_range = lambda w: str(clus['batch_rxn_sols']*int(w.parallel)) + '_' + str(clus['batch_rxn_sols']*(int(w.parallel)+1))
    log:
        "logs/rxn_enum_{condition}_{parallel}.log"
    shell:
        "python utilities_cluster/cluster_rxn_enum.py -c {wildcards.condition} -p {wildcards.parallel} -r {params.rxn_range}"

rule imat:
    input:
        config['model'],
        config['reaction_weights']
    output:
        outpath+"imat_solution.csv"
    log:
        "logs/imat.log"
    shell:
        "python dexom_python/imat_functions.py -m %s -r %s" %
        (config['model'], config['reaction-weights'])

rule approach_1:

rule approach_2

rule approach_3:

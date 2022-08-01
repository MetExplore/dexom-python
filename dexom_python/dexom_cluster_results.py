from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def analyze_dexom_cluster_results(in_folder, out_folder, approach=1, filenums=100):
    """

    Parameters
    ----------
    in_folder: folder containing dexom results
    out_folder: folder in which output files will be saved
    approach: which parallelization approach was used (1, 2, or 3, see enum_functions/enumeration for details)
    filenums: number of parallel dexom threads that were run

    Returns
    -------

    """
    output_file = []
    # concatenating all .out files from the cluster
    if approach == 1 or approach == 3:
        fileout = 'dex1out'
        fileerr = 'dex1err'
        with open(out_folder+'all_outs.txt', 'w+') as outfile:
            for i in range(filenums):
                fname = in_folder+fileout+str(i)+'.out'
                with open(fname) as infile:
                    outfile.write(infile.read())
        with open(out_folder+'all_errs.txt', 'w+') as outfile:
            for i in range(filenums):
                fname = in_folder+fileerr+str(i)+'.out'
                with open(fname) as infile:
                    outfile.write(infile.read())
    elif approach == 2:
        outfiles = Path(in_folder).glob('*out*.out')
        errfiles = Path(in_folder).glob('*err*.out')
        with open(out_folder + 'all_outs.txt', 'w+') as outfile:
            for f in outfiles:
                with open(str(f)) as infile:
                    outfile.write(infile.read())
        with open(out_folder + 'all_errs.txt', 'w+') as outfile:
            for f in errfiles:
                with open(str(f)) as infile:
                    outfile.write(infile.read())

    #concatenating & analyzing rxn_enum results
    output_file.append('looking at rxn_enum')
    print(output_file[-1])
    all_rxn = []
    if approach == 3:
        rxn = pd.read_csv(in_folder+'rxn_enum_solutions.csv', index_col=0)
    else:
        for i in range(filenums):
            try:
                if approach == 1:
                    filename = in_folder + 'rxn_enum_%i_solutions.csv' % i
                elif approach == 2:
                    filename = Path(in_folder).glob('div_enum_%i_*_solutions.csv' % i)
                    filename = str(list(filename)[0])
                rxn = pd.read_csv(filename, index_col=0)
                all_rxn.append(rxn)
            except FileNotFoundError:
                pass  # if a file is absent, ignore it
        rxn = pd.concat(all_rxn, ignore_index=True)
    if approach == 1 or approach == 3:
        unique = len(rxn.drop_duplicates())
        output_file.append('There are %i unique solutions and %i duplicates' % (unique, len(rxn) - unique))
        print(output_file[-1])
        fulltime = 0
        counter = 0
        with open(out_folder+'all_outs.txt', 'r') as file:
            for line in file:
                line = line.split()
                try:
                    fulltime += float(line[0])
                    counter += 1
                except (ValueError, IndexError):
                    pass  # ignore lines that are empty or don't begin with a number
        if counter != 0:
            output_file.append('Total computation time: %i s' % int(fulltime))
            print(output_file[-1])
            output_file.append('Average time per iteration: %.2f s' % (fulltime*2/counter))
            print(output_file[-1])
    if approach == 2:
        all_res = []
        for i in range(filenums):
            try:
                filename = Path(in_folder).glob('div_enum_%i_*_results.csv' % i)
                filename = str(list(filename)[0])
                res = pd.read_csv(filename, index_col=0)
                all_res.append(res)
            except FileNotFoundError:
                pass
        rxn_res = pd.concat(all_res, ignore_index=True)
        rxn_res.to_csv(out_folder + 'all_rxn_enum_res.csv')

    # concatenating & analyzing diversity_enum results
    output_file.append('looking at diversity_enum')
    print(output_file[-1])
    all_res = []
    all_sol = []
    if approach == 1 or approach == 3:
        for i in range(filenums):
            try:
                solname = in_folder + 'div_enum_%i_solutions.csv' % i
                resname = in_folder + 'div_enum_%i_results.csv' % i
                sol = pd.read_csv(solname, index_col=0)
                res = pd.read_csv(resname, index_col=0)
                all_sol.append(sol)
                all_res.append(res)
            except FileNotFoundError:
                pass
    elif approach == 2:
        solname = Path(in_folder).glob('div_enum2021*_solutions.csv')
        all_sol = [pd.read_csv(str(x), index_col=0) for x in solname]
        resname = Path(in_folder).glob('div_enum2021*_results.csv')
        all_res = [pd.read_csv(str(x), index_col=0) for x in resname]
    sol = pd.concat(all_sol, ignore_index=True)
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(out_folder+'all_divenum_res.csv')
    unique = len(sol.drop_duplicates())
    output_file.append('There are %i unique solutions and %i duplicates' % (unique, len(sol)-unique))
    print(output_file[-1])
    time = res['time'].cumsum()
    output_file.append('Total computation time: %i s' % time.iloc[-1])
    print(output_file[-1])
    output_file.append('Average time per iteration: %.2f s' % (time.iloc[-1]/len(sol)))
    print(output_file[-1])

    plt.clf()
    fig = res.sort_values('selected reactions').reset_index(drop=True)['selected reactions'].plot().get_figure()
    fig.savefig(out_folder+'all_divenum_selected_reactions_ordered.png')
    # analyzing total results
    output_file.append('total result')
    print(output_file[-1])
    full = pd.concat([rxn, sol], ignore_index=True)
    unique = len(full.drop_duplicates())
    output_file.append('There are %i unique solutions and %i duplicates' % (unique, len(full)-unique))
    print(output_file[-1])
    rxn = rxn.drop_duplicates(ignore_index=True)
    rxn.to_csv(out_folder+'all_rxnenum_sols.csv')
    sol = sol.drop_duplicates(ignore_index=True)
    sol.to_csv(out_folder+'all_divenum_sols.csv')
    full = full.drop_duplicates(ignore_index=True)
    full.to_csv(out_folder+'all_dexom_sols.csv')
    with open(out_folder+'output.txt', 'w+') as file:
        for x in output_file:
            file.write(x+'\n')
    return full


if __name__ == '__main__':
    description = 'Compiles and analyzes results from parallel DEXOM'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--in_path', default='', help='Path in which the cluster results were saved')
    parser.add_argument('-o', '--out_path', default='', help='Path in which to save compiled results')
    parser.add_argument('-n', '--filenums', type=int, default=100, help='number of parallel threads')
    parser.add_argument('-a', '--approach', type=int, default=1,
                        help='which parallelization approach was used (1 by default)')
    args = parser.parse_args()

    analyze_dexom_cluster_results(in_folder=args.in_path, out_folder=args.out_path, approach=args.approach,
                                  filenums=args.filenums)

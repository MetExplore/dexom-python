import argparse
from dexom_python.enum_functions.enumeration import combine_binary_solutions_and_fluxes


def _main():
    """
    This function is called when you run this script from the commandline.
    Compiles binary enumeration solutions from a given folder
    Use --help to see commandline parameters
    """
    description = 'Compiles binary enumeration solutions from a given folder'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--sol_path', help='folder containing enumeration solutions')
    parser.add_argument('-o', '--out_path', default='', help='path to which the combined solutions will be saved')
    parser.add_argument('-p', '--pattern', default='*solutions.csv', help='naming pattern of the solution files')
    args = parser.parse_args()
    sols = combine_binary_solutions_and_fluxes(sol_path=args.sol_path, solution_pattern=args.pattern, out_path=args.out_path)
    sols.sum()._set_name('frequency').to_csv(args.out_path+'activation_frequency_reactions.csv')
    return True


if __name__ == '__main__':
    _main()

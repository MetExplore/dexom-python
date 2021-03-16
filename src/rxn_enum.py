
import argparse
from cobra.io import load_json_model
from models import load_reaction_weights, read_solution
from iMAT import imat


if __name__=="__main__":
    description = "Performs the reaction enumeration algorithm"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", help="Metabolic model in json format")
    parser.add_argument("-r", "--reaction_weights", default={}, help="Reaction weights in csv format (first row: reaction names, second row: weights)")
    parser.add_argument("-i", "--initial_solution", default=None, help="initial imat solution in .txt format")
    parser.add_argument("-e", "--epsilon", default=1., help="Activation threshold for highly expressed reactions")
    parser.add_argument("-t", "--threshold", default=1e-1, help="Activation threshold for all reactions")
    parser.add_argument("-o", "--output", default="rxn_solution.txt", help="Name of the output file")

    args = parser.parse_args()

    model = load_json_model(args.model)

    reaction_weights = {}
    if args.reaction_weights:
        reaction_weights = load_reaction_weights(args.reaction_weights)

    initial_solution, initial_binary = read_solution(args.initial_solution)

    # solution = imat(model, reaction_weights, args.epsilon, args.threshold)

    # write_solution(solution, args.threshold, args.output)

"""
Main file for training model, visualizing results and preprocessing
"""

import argparse
from matplotlib.pyplot import fill
from rnn import run_model
from visualization_utilities import fill_with_model
from preprocess_utilities import preprocess

params_of_interest = ['chlorophyll']



def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_bool", action="store_true")
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--raw_data_file", type=str, default='data/merged_sst_ice_chl_par_2003.RDS')
    parser.add_argument("--mapping_file", type=str, default='data/Bering_full_grid_lookup_no_goa.RDS')
    parser.add_argument("--inputs_path", type=str, default='preprocessed-data/2003_50_244_1_1_chlorophyll_inputs.npy')
    parser.add_argument("--labels_path", type=str, default='preprocessed-data/2003_50_244_1_1_chlorophyll_labels.npy')
    parser.add_argument("--visualization_path", type=str, default='imgs/filled_with_model.gif')
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    if args.preprocess_bool:
            preprocess(args.raw_data_file, args.mapping_file, params_of_interest, 50, 140, args.t, args.n, 
            'preprocessed-data/test_inputs.npy', 'preprocessed-data/test_lables.npy')
    if args.train:
        trained_model = run_model(args.inputs_path, args.labels_path, args.n)
        fill_with_model(trained_model, args.raw_data_file, args.mapping_file, 50, 140, args.t, args.n, args.visualization_path)





if __name__ == "__main__":
    args = parseArguments()
    main(args)



"""
Main file for training model, visualizing results and preprocessing
"""

from rnn import run_model
from visualization_utilities import fill_with_model
from preprocess_utilities import preprocess, generate_output_paths
from argparser import parse_args

preprocess_bool = False
train_model = True

mapping_file = 'data/Bering_full_grid_lookup_no_goa.RDS'

if __name__ == "__main__":
    # uncomment below for use with command line
    args = parse_args()
    year = args.year
    start_day = args.start_day
    end_day = args.end_day
    time_window = args.time_window
    num_neighbors = args.num_neighbors
    data_types = args.data_types
    VERBOSE = args.verbose

    input_data_file = f"data/merged_sst_ice_chl_par_{year}.RDS"
    inputs_file_path, labels_file_path = generate_output_paths(year, start_day, end_day, time_window, num_neighbors, data_types)

    if preprocess_bool:
        preprocess(input_data_file, mapping_file, data_types, start_day, end_day, time_window, num_neighbors, inputs_file_path, labels_file_path)
    if train_model:
        trained_model = run_model(inputs_file_path, labels_file_path)
        fill_with_model(trained_model, input_data_file, mapping_file, start_day, end_day, time_window, num_neighbors)

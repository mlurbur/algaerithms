"""
Main file for training model, visualizing results and preprocessing
"""

from matplotlib.pyplot import fill
from rnn import run_model
from visualization_utilities import fill_with_model
from preprocess_utilities import preprocess

preprocess_bool = True
train = False


raw_data_file = 'data/merged_sst_ice_chl_par_2003.RDS'
mapping_file = 'data/Bering_full_grid_lookup_no_goa.RDS'
params_of_interest = ['chlorophyll']
inputs_path = 'preprocessed-data/2003_50_244_1_1_chlorophyll_inputs.npy'
labels_path = 'preprocessed-data/2003_50_244_1_1_chlorophyll_labels.npy'









if __name__ == "__main__":
    if preprocess_bool:
        preprocess(raw_data_file, mapping_file, params_of_interest, 50, 140, 3, 1, 
            'preprocessed-data/test_inputs.npy', 'preprocessed-data/test_lables.npy')
    if train:
        trained_model = run_model(inputs_path, labels_path)
        fill_with_model(trained_model, raw_data_file, mapping_file, 50, 140, 3, 1)



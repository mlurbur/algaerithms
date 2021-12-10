"""
Main file for training model, visualizing results and preprocessing

Usage:
python main.py RNN 2003 50 244 2 1 chlorophyll -r
"""

from tensorflow.python.training.tracking import base
from visualization_utilities import fill_with_model, compare_to_baseline
from preprocess_utilities import preprocess, generate_output_paths, split_data
from argparser import parse_args
from model import RNN, FFN
import tensorflow as tf
import numpy as np

def train(model, train_inputs, train_labels):
    """
    Trains the model for one epoch. Returns the average batch MSE.
    """
    total_loss = 0
    num_batches = 0
    for i in range(0, len(train_inputs), model.batch_size):
        batched_inputs = train_inputs[i : i + model.batch_size]
        batched_labels = train_labels[i : i + model.batch_size]
        with tf.GradientTape() as tape:
            pred = tf.squeeze(model.call(batched_inputs))
            loss = model.loss_mse(pred, batched_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
        num_batches += 1
    return total_loss / num_batches

def test(model, test_inputs, test_labels):
    """
    Tests the model. Returns the average batch MAPE and MSE.
    """
    total_loss = 0
    num_batches = 0
    num_test_examples = test_labels.shape[0]
    all_pred = np.zeros(num_test_examples)
    for i in range(0, len(test_inputs), model.batch_size):
        batched_inputs = test_inputs[i : i + model.batch_size]
        batched_labels = test_labels[i : i + model.batch_size]
        pred = tf.squeeze(model.call(batched_inputs))
        mse = model.loss_mse(pred, batched_labels)
        all_pred[i:i+model.batch_size] = pred
        total_loss += mse
        num_batches += 1
    return model.loss_mape(test_labels, all_pred), total_loss / num_batches

def baseline_mape(inputs, labels, num_neighbors):
    chlorophyll_values = inputs[:, -1, np.math.floor(((2 * num_neighbors + 1)**2)/2)]
    # avg_chlorophyll_value = np.mean(chlorophyll_values)
    mape = 100 * tf.math.reduce_mean(tf.math.abs((chlorophyll_values - labels) / labels))
    return mape


def run_model(inputs_path_list, labels_path_list, num_neighbors, model_type):
    # extra preprocessing to aggregate all data files
    all_inputs = []
    all_labels = []
    for inputs_path, labels_path in zip(inputs_path_list, labels_path_list):
        all_inputs.append(np.load(inputs_path))
        all_labels.append(np.load(labels_path))
    inputs = np.concatenate(all_inputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    # instantiate the model
    model = None
    if model_type == "RNN":
        model = RNN(inputs.shape[1])
    if model_type == "FFN":
        model = FFN(inputs.shape[1])
    assert(model is not None)
    # train model for model.epochs epochs
    print("Training the", model_type, "...")
    for i in range(model.epochs):
        train_inputs, train_labels, test_inputs, test_labels = split_data(inputs, labels)
        train_mse = train(model, train_inputs, train_labels)
        test_mape, test_mse = test(model, test_inputs, test_labels)
        base_mape = baseline_mape(test_inputs, test_labels, num_neighbors)
        print(f"Epoch: {i+1} | Train MSE: {tf.math.round(train_mse)}; Test MSE: {tf.math.round(test_mse)}; Test MAPE: {tf.math.round(test_mape)}%")
    # return the trained model
    return model

mapping_file = 'data/Bering_full_grid_lookup_no_goa.RDS'

def main():
    args = parse_args()
    year = args.year
    start_day = args.start_day
    end_day = args.end_day
    time_window = args.time_window
    num_neighbors = args.num_neighbors
    data_types = args.data_types
    model = args.model
    PREPROCESS = args.preprocess
    RUN = args.run
    input_data_file = f"data/merged_sst_ice_chl_par_{year}.RDS"
    inputs_file_path, labels_file_path = generate_output_paths(year, start_day, end_day, time_window, num_neighbors, data_types)
    visualization_path = f'imgs/data_filled_with_{model}.gif'

    if PREPROCESS:
        preprocess(input_data_file, mapping_file, data_types, start_day, end_day, time_window, num_neighbors, inputs_file_path, labels_file_path)
    if RUN:
        trained_model = run_model([inputs_file_path], [labels_file_path], num_neighbors, model)
        # doesn't make sense to use model MAPE here since it will be seeing some of the training set
        # I think that the baseline is still pretty valid
        _, baseline_mape = compare_to_baseline(trained_model, input_data_file, mapping_file, start_day, end_day, time_window, num_neighbors)
        print(f"Baseline mape: {baseline_mape}%")
        fill_with_model(trained_model, input_data_file, mapping_file, start_day, end_day, time_window, num_neighbors, visualization_path)

if __name__ == "__main__":
    main()

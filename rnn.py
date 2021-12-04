import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError

import timeit


class Model(tf.keras.Model):
    def __init__(self, time_step):
        """
        Initilize all the hyper parameters for the LSTM model
        """

        super(Model, self).__init__()

        self.hidden_size = 256
        self.time_step = time_step
        self.embedding_size = 128
        self.batch_size = 256
        self.learning_rate = 0.01
        self.epochs = 5

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.lstm_layer1 = tf.keras.layers.LSTM(1028, return_sequences=True, return_state=False)
        self.lstm_layer2 = tf.keras.layers.LSTM(1028, return_sequences=True, return_state=False)
        self.lstm_layer3 = tf.keras.layers.LSTM(1028, return_sequences=False, return_state=False)
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)
        

    def call(self, inputs, initial_state):
        """
        Forward pass *t = window size

        Inputs - shape = (N, t, (2n+1)(2n+1)-1 * num data types)
        Initial state - Fake input

        Returns
        """

        lstm_output1 = self.lstm_layer1(inputs)
        lstm_output2 = self.lstm_layer2(lstm_output1)
        lstm_output3 = self.lstm_layer3(lstm_output2)
        hidden1 = self.dense1(lstm_output3)
        hidden2 = self.dense2(hidden1)
        predictions = self.dense3(hidden2)

        return predictions

    def loss(self, pred, labels):
        """
        Calculates MSE loss
        """
        mse = MeanSquaredError()
        return mse(labels, pred)
    
    def accuracy(self, labels, predicted):
        abs_error = np.absolute((labels - predicted)/predicted)
        return 100 * np.mean(abs_error)

def reshape_inputs_and_labels(inputs, labels, window_size):
    """
    window size = time step
    """
    number_of_inputs = len(inputs)//window_size
    reshaped_inputs = np.zeros((number_of_inputs, window_size),  dtype=np.int64)
    reshaped_labels = np.zeros((number_of_inputs, window_size),  dtype=np.int64)
    for i in range(number_of_inputs):
        reshaped_inputs[i, :] = inputs[i * window_size: (i + 1) * window_size]
        reshaped_labels[i, :] = labels[(i * window_size): ((i + 1) * window_size)]

    return reshaped_inputs, reshaped_labels


def train(model, train_inputs, train_labels):

    # train_x, train_y = reshape_inputs_and_labels(train_inputs, train_labels, model.time_step)

    total_loss = 0
    step = 0

    for i in range(0, len(train_inputs), model.batch_size):
        batch_x = train_inputs[i:i+model.batch_size]
        batch_y = train_labels[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            pred = model.call(batch_x, None)
            loss = model.loss(pred, batch_y)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
        step += 1
    
    return total_loss/step



def test(model, test_inputs, test_labels):

    test_x, test_y = reshape_inputs_and_labels(test_inputs, test_labels, model.time_step)

    curr_loss = 0
    step = 0

    for i in range(0, len(test_x), model.batch_size):
        batch_x = test_x[i:i+model.batch_size]
        batch_y = test_y[i:i+model.batch_size]
        probs, final_state = model.call(batch_x, None)
        loss = model.loss(probs, batch_y)
        step+=1
        curr_loss+=loss

    return np.exp(curr_loss/step)


def generate_inputs_and_labels(data):
    return data[:-1], data[1:]

def main():
    train_data = np.load("data.npy")
    ground_truth = np.load("gt.npy")

    model = Model(train_data.shape[1])

    print("starting train")
    for i in range(model.epochs):
        loss = train(model, train_data, ground_truth)
        print("epoch: {} loss: {}".format(i + 1, loss))



if __name__ == '__main__':
    main()

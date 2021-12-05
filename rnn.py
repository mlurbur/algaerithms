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

        self.hidden_size = 500
        self.rnn_units = 164
        self.time_step = time_step
        self.batch_size = 200
        self.learning_rate = 0.001
        self.epochs = 25

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.lstm_layer1 = tf.keras.layers.LSTM(self.rnn_units, return_sequences=True, return_state=False)
        # self.lstm_layer2 = tf.keras.layers.LSTM(1028, return_sequences=True, return_state=False)
        self.lstm_layer3 = tf.keras.layers.LSTM(self.rnn_units, return_sequences=False, return_state=False)
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
        # lstm_output2 = self.lstm_layer2(lstm_output1)
        lstm_output3 = self.lstm_layer3(lstm_output1)
        hidden1 = self.dense1(lstm_output3)
        hidden2 = self.dense2(hidden1)
        predictions = self.dense3(hidden2)

        # get the final predicted values (i think)
        pred = predictions[:,0]
        return pred

    def loss(self, pred, labels):
        """
        Calculates MSE loss
        """
        mse = MeanSquaredError()
        return mse(labels, pred)

    def loss_mape(self, pred, labels):
        """
        Calculates MAPE loss
        """
        abs_error = tf.math.abs((pred - labels)/labels)
        return 100 * tf.math.reduce_mean(abs_error)
    
    def accuracy(self, labels, predicted):
        abs_error = np.absolute((predicted - labels)/labels)
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

    total_loss = 0
    step = 0

    for i in range(0, len(train_inputs), model.batch_size):
        batch_x = train_inputs[i:i+model.batch_size]
        batch_y = train_labels[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            pred = model.call(batch_x, None)
            loss = model.loss_mape(pred, batch_y)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
        step += 1
    
    return total_loss/step


def test(model, test_inputs, test_labels):

    curr_loss = 0
    step = 0
    all_pred = np.zeros(test_labels.shape[0])

    # for i in range(0, len(test_inputs), model.batch_size):
    #     batch_x = test_inputs[i:i+model.batch_size]
    #     batch_y = test_labels[i:i+model.batch_size]
    #     pred = model.call(batch_x, None)
    #     loss = model.loss(pred, batch_y)
    #     all_pred[i:i+model.batch_size] = pred
    #     step+=1
    #     curr_loss+=loss
    pred = model.call(test_inputs, None)
    loss = model.loss_mape(pred, test_labels)

    return model.accuracy(test_labels, pred), loss

def average(inputs, labels):
    chlor = inputs[:,:,:9]
    avg = np.mean(chlor, axis=1)
    acc = 100 * np.mean(np.absolute((avg - labels)/labels))
    return acc


def split_data(inputs, labels):
    shuffler = np.random.permutation(len(labels))
    labels_shuffled = labels[shuffler]
    inputs_shuffled = inputs[shuffler]
    split = (len(labels_shuffled) * 8)//10
    train_inputs = inputs_shuffled[:split]
    test_inputs = inputs_shuffled[split + 1:]
    train_labels = labels_shuffled[:split]
    test_labels = labels_shuffled[split + 1:]

    return train_inputs, train_labels, test_inputs, test_labels

def main():
    train_data = np.load("data.npy")
    ground_truth = np.load("gt.npy")

    model = Model(train_data.shape[1])

    print("starting train")
    for i in range(model.epochs):
        train_inputs, train_labels, test_inputs, test_labels = split_data(train_data, ground_truth)
        train_loss = train(model, train_inputs, train_labels)
        test_acc, test_loss = test(model, test_inputs, test_labels)
        print("epoch: {} train_loss: {}, test_loss: {}, test_err: {}".format(i + 1, tf.math.round(train_loss), tf.math.round(test_loss), tf.math.round(test_acc)))


def test_acc():
    model = Model(1)

    pred = np.array([1,1,1])
    label = np.array([1,1,1])
    assert model.accuracy(label, pred) == 0.0, "failed"

    pred = np.array([0,0,0])
    label = np.array([1,1,1])
    assert model.accuracy(label, pred) == 100.0, "failed"

    pred = np.array([-1,-1,-1])
    label = np.array([1,1,1])
    assert model.accuracy(label, pred) == 200.0, "failed"

    print("all tests passed")

if __name__ == '__main__':
    # test_acc()
    main()

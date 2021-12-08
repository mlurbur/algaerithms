import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Model(tf.keras.Model):
    def __init__(self, time_step):
        """
        Initialize all the hyper parameters for the LSTM model
        """

        super(Model, self).__init__()

        self.hidden_size = 512
        self.rnn_units = 1024
        self.time_step = time_step
        self.batch_size = 200
        self.learning_rate = 0.001
        self.epochs = 2

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.lstm_layer1 = tf.keras.layers.LSTM(self.rnn_units*2, return_sequences=True, return_state=False)
        self.lstm_layer2 = tf.keras.layers.LSTM(self.rnn_units, return_sequences=True, return_state=False)
        self.lstm_layer3 = tf.keras.layers.LSTM(self.rnn_units, return_sequences=False, return_state=False)
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.hidden_size//2, activation='relu')
        self.dense4 = tf.keras.layers.Dense(self.hidden_size//4, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1)
        

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
        logit1 = self.dense1(lstm_output3)
        logit2 = self.dense2(logit1)
        logit3 = self.dense3(logit2)
        logit4 = self.dense4(logit3)
        predictions = self.dense5(logit4)

        # get the final predicted values
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
        """
        Calculates MAPE
        """
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

    for i in range(0, len(test_inputs), model.batch_size):
        batch_x = test_inputs[i:i+model.batch_size]
        batch_y = test_labels[i:i+model.batch_size]
        pred = model.call(batch_x, None)
        loss = model.loss(pred, batch_y)
        all_pred[i:i+model.batch_size] = pred
        step+=1
        curr_loss+=loss

    return model.accuracy(test_labels, all_pred), curr_loss/step

def average(inputs, labels, n):
    chlor = inputs[0,0,:(n*2 + 1)**2]
    avg = np.mean(chlor)
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

def plot_training(train_loss, test_loss, test_acc, epochs):
    x_values = np.arange(1, epochs + 1)
    gs = gridspec.GridSpec(6, 1)

    plt.figure()
    plt.subplot(gs[0, 0])
    plt.plot(x_values, test_acc * 100)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.title('Test Accuracy')
    
    plt.subplot(gs[2, 0])
    plt.plot(x_values, test_loss * 100)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Test Loss')
    
    plt.subplot(gs[4, 0])
    plt.plot(x_values, train_loss * 100)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Train Loss')

    plt.savefig('some_name.png')
    print('Finished Plotting')

def run_model(inputs_path_list, labels_path_list, n):
    inputs_list = []
    labels_list = []
    for i, j in zip(inputs_path_list, labels_path_list):
        inputs_list.append(np.load(i))
        labels_list.append(np.load(j))

    data = np.concatenate(inputs_list,axis=0)
    labels = np.concatenate(labels_list,axis=0)
    model = Model(data.shape[1])

    print("starting train")
    track_train_loss = []
    track_test_acc = []
    track_test_loss = []
    for i in range(model.epochs):
        train_inputs, train_labels, test_inputs, test_labels = split_data(data, labels)
        train_loss = train(model, train_inputs, train_labels)
        test_acc, test_loss = test(model, test_inputs, test_labels)
        print("epoch: {} train_loss: {}, test_loss: {}, test_err: {}, baseline_err: {}".format(i + 1, 
            tf.math.round(train_loss), tf.math.round(test_loss), tf.math.round(test_acc), tf.math.round(average(test_inputs, test_labels, n))))
        track_test_acc.append(test_acc)
        track_test_loss.append(test_loss)
        track_train_loss.append(train_loss)

    plot_training(track_train_loss, track_test_loss, track_test_acc, model.epochs)


    

    return model

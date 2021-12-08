import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

class RNN(Model):
    def __init__(self, time_step):
        super(RNN, self).__init__()

        # define model hyperparameters
        self.time_step = time_step
        self.hidden_size = 500
        self.rnn_units = 164

        self.batch_size = 200
        self.learning_rate = 1e-3
        self.epochs = 25

        # initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # rnn architecture
        self.network = Sequential()
        self.network.add(LSTM(self.rnn_units, return_sequences=True, return_state=False))
        # self.network.add(LSTM(self.rnn_units, return_sequences=False, return_state=False))
        self.network.add(Dense(self.hidden_size, activation='relu'))
        self.network.add(Dense(self.hidden_size, activation='relu'))
        self.network.add(Dense(1))

    def call(self, inputs):
        """
        Runs the forward pass on batched inputs ((N, t, ((2n+1)^2)-1 * len(data_types))).
        """
        return self.network(inputs)

    def loss_mse(self, pred, labels):
        """
        Calculates the MSE loss.
        """
        mse = MeanSquaredError()
        return mse(labels, pred)

    def loss_mape(self, pred, labels):
        """
        Calculates the MAPE (mean absolute percentage error) loss.
        """
        abs_error = tf.math.abs((pred - labels) / labels)
        return 100 * tf.math.reduce_mean(abs_error)

class FFN(Model):
    def __init__(self, time_step):
        super(FFN, self).__init__()

        # define model hyperparameters
        self.time_step = time_step
        self.hidden_size = 400

        self.batch_size = 200
        self.learning_rate = 1e-3
        self.epochs = 25

        # initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # feed-forward network architecture
        self.network = Sequential()
        self.network.add(Dense(self.hidden_size, activation='relu'))
        self.network.add(Dense(self.hidden_size, activation='relu'))
        self.network.add(Dense(self.hidden_size, activation='relu'))
        self.network.add(Dense(1))

    def call(self, inputs):
        """
        Runs the forward pass on batched inputs ((N, t, ((2n+1)^2)-1 * len(data_types))).
        """
        return self.network(inputs)

    def loss_mse(self, pred, labels):
        """
        Calculates the MSE loss.
        """
        mse = MeanSquaredError()
        return mse(labels, pred)

    def loss_mape(self, pred, labels):
        """
        Calculates the MAPE (mean absolute percentage error) loss.
        """
        abs_error = tf.math.abs((pred - labels) / labels)
        return 100 * tf.math.reduce_mean(abs_error)

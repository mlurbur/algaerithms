import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import gen_data, fill_missing

import timeit


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        Initilize all the hyper parameters for the LSTM model
        """

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 128
        self.batch_size = 128
        self.learning_rate = 0.01
        self.epochs = 1

        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=0.1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.lstm_layer1 = tf.keras.layers.LSTM(1028, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(self.vocab_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='softmax')


    def call(self, inputs, initial_state):
        """
        Forward pass *t = window size

        Inputs - shape = (N, t, (2n+1)(2n+1)-1 * num data types)
        Initial state - Fake input

        Returns
        """

        embedding = tf.nn.embedding_lookup(self.E, inputs)
        whole_seq_output, final_memory_sate, final_carry_state = self.lstm_layer1(embedding)
        outputs = self.dense1(whole_seq_output)

        return outputs,[final_memory_sate, final_carry_state]

    def loss(self, probs, labels):

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probs,from_logits=False,))

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

    train_x, train_y = reshape_inputs_and_labels(train_inputs, train_labels, model.window_size)

    for i in range(0, len(train_x), model.batch_size):
        batch_x = train_x[i:i+model.batch_size]
        batch_y = train_y[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            probs, final_state = model.call(batch_x, None)
            loss = model.loss(probs, batch_y)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):

    test_x, test_y = reshape_inputs_and_labels(test_inputs, test_labels, model.window_size)

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
    original_data = np.load("data.npy")
    time_step = 10
    batch_size = 128
    data = gen_data(original_data, fill_missing(original_data), time_step, batch_size)

    print(data.shape)

    model = Model(len(vocab_dict))

    for i in range(model.epochs):
        print("epoch {}".format(i + 1))
        train(model, train_inputs, train_labels)

    perplexity = test(model, test_inputs, test_labels)
    print("test perplexity = {}".format(perplexity))

    print("sample_n=10")
    generate_sentence("he", 6, vocab_dict, model, sample_n=10)
    print("sample_n=20")
    generate_sentence("he", 6, vocab_dict, model, sample_n=20)


if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data

import timeit


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
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
        self.dense1 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')


    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        whole_seq_output, final_memory_sate, final_carry_state = self.lstm_layer1(embedding)
        probs = self.dense1(whole_seq_output)

        return probs,[final_memory_sate, final_carry_state]

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probs,from_logits=False,))

def reshape_inputs_and_labels(inputs, labels, window_size):
    number_of_inputs = len(inputs)//window_size
    reshaped_inputs = np.zeros((number_of_inputs, window_size),  dtype=np.int64)
    reshaped_labels = np.zeros((number_of_inputs, window_size),  dtype=np.int64)
    for i in range(number_of_inputs):
        reshaped_inputs[i, :] = inputs[i * window_size: (i + 1) * window_size]
        reshaped_labels[i, :] = labels[(i * window_size): ((i + 1) * window_size)]

    return reshaped_inputs, reshaped_labels


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
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
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
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


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def generate_inputs_and_labels(data):
    return data[:-1], data[1:]

def main():
    train_file = "../../data/train.txt"
    test_file = "../../data/test.txt"
    train_data, test_data, vocab_dict = get_data(train_file, test_file)
    train_inputs, train_labels = generate_inputs_and_labels(train_data)
    test_inputs, test_labels = generate_inputs_and_labels(test_data)

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

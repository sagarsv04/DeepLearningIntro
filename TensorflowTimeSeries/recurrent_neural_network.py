from __future__ import print_function, division
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


# os.getcwd()
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\TensorflowTimeSeries')


# hyperparams
num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


# the input is basically a random binary vector. The output will be the “echo” of the input, shifted echo_step steps to the right.
def generate_data():
    # 0,1, 50K samples, 50% chance each chosen
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    # shift 3 steps to the left
    y = np.roll(x, echo_step)
    # padd beginning 3 values with 0
    y[0:echo_step] = 0
    # Gives a new shape to an array without changing its data.
    # The reshaping takes the whole dataset and puts it into a matrix,
    # that later will be sliced up into these mini-batches.
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return x, y


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


def run_rnn():

    data = generate_data()
    # datatype, shape (5, 15) 2D array or matrix, batch size shape for later
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

    # and one for the RNN state, 5,4
    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    # randomly initialize weights
    W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
    # anchor, improves convergance, matrix of 0s
    b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

    W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

    # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
    # so a bunch of arrays, 1 batch per time step
    inputs_series = tf.unstack(batchX_placeholder, axis=1)
    labels_series = tf.unstack(batchY_placeholder, axis=1)

    # Forward pass
    # state placeholder
    current_state = init_state
    # series of states through time
    states_series = []

    # for each set of inputs
    # forward pass through the network to get new state value
    # store all states in memory
    for current_input in inputs_series:
        # current_input = inputs_series[0]
        # format input
        current_input = tf.reshape(current_input, [batch_size, 1])
        # mix both state and input data
        input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
        # perform matrix multiplication between weights and input, add bias
        # squash with a nonlinearity, for probabiolity value
        next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
        # store the state in memory
        states_series.append(next_state)
        # set current state to next one
        current_state = next_state

    # second part of forward pass
    # logits short for logistic transform
    logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
    # apply softmax nonlinearity for output probability
    predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

    # measure loss, calculate softmax again on logits, then compute cross entropy
    # measures the difference between two probability distributions
    # this will return A Tensor of the same shape as labels and of the same type as logits
    # with the softmax cross entropy loss.
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
    # computes average, one value
    total_loss = tf.reduce_mean(losses)
    # great paper http://seed.ucsd.edu/mediawiki/images/6/6a/Adagrad.pdf
    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        plt.ion() # interactive mode
        plt.figure() # initialize the figure
        plt.show() # show the graph
        loss_list = [] # to show the loss decrease

        for epoch_idx in range(num_epochs):
            # generate data at eveery epoch, batches run in epochs
            x, y = data
            # initialize an empty hidden state
            _current_state = np.zeros((batch_size, state_size))

            print("New data, epoch", epoch_idx)
            # each batch
            for batch_idx in range(num_batches):
                # These layers will not be unrolled to the beginning of time,
                # that would be too computationally expensive, and are therefore truncated
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:,start_idx:end_idx]
                batchY = y[:,start_idx:end_idx]

                # run the computation graph, give it the values
                # we calculated earlier
                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series],
                    feed_dict={batchX_placeholder:batchX, batchY_placeholder:batchY, init_state:_current_state})

                loss_list.append(_total_loss)

                if batch_idx%100 == 0:
                    print("Step",batch_idx, "Loss", _total_loss)
                    plot(loss_list, _predictions_series, batchX, batchY)

    plt.ioff()
    plt.show()

    return 0



def main():

    run_rnn()

    return 0


if __name__ == '__main__':
    main()

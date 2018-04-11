import pandas as pd              # A beautiful library to help us work with data as tables
import numpy as np               # So we can use number matrices. Both pandas and TensorFlow need it.
import matplotlib.pyplot as plt  # Visualize the things
import tensorflow as tf          # Fire from the gods
import os

# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TensorFlowClassification')

house_data_path = "./house_prices.csv"


def classify_house_prices():

    # Load our dataset into a dataframe
    dataframe = pd.read_csv(house_data_path)
    # Remove columns we don't care about
    dataframe = dataframe[["area", "bathrooms"]]
    # houses liked list generated randomly
    liked_list = np.random.randint(0,2,len(dataframe))
    # houses liked denoted by Y1
    dataframe["Y1"] = liked_list
    # Y2 is negation of Y1
    # astype(int) turn TRUE/FALSE values into 1/0
    dataframe["Y2"] = (~(dataframe.Y1==1)).astype(int)

    inputX = dataframe[['area', 'bathrooms']].as_matrix()
    inputY = dataframe[["Y1", "Y2"]].as_matrix()

    # Parameters
    learning_rate = 0.000001
    training_epochs = 2000
    display_step = 50
    n_samples = inputY.size

    # Okay TensorFlow, we'll feed you an array of examples. Each example will
    # be an array of two float values (area, and number of bathrooms).
    # "None" means we can feed you any number of examples
    # Notice we haven't fed it the values yet
    x = tf.placeholder(tf.float32, [None, 2])
    # Maintain a 2 x 2 float matrix for the weights that we'll keep updating
    # through the training process (make them all zero to begin with)
    W = tf.Variable(tf.zeros([2, 2]))
    # Also maintain two bias values
    b = tf.Variable(tf.zeros([2]))
    # The first step in calculating the prediction would be to multiply
    # the inputs matrix by the weights matrix then add the biases
    y_values = tf.add(tf.matmul(x, W), b)
    # Then we use softmax as an "activation function" that translates the
    # numbers outputted by the previous layer into probability form
    y = tf.nn.softmax(y_values)
    # For training purposes, we'll also feed you a matrix of labels
    y_ = tf.placeholder(tf.float32, [None,2])

    # Cost function: Mean squared error
    cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize variabls and tensorflow session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(training_epochs):
        # Take a gradient descent step using our inputs and labels
        sess.run(optimizer, feed_dict={x: inputX, y_: inputY})

        # That's all! The rest of the cell just outputs debug messages.
        # Display logs per epoch step
        if (i) % display_step == 0:
            cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
             #, \"W=", sess.run(W), "b=", sess.run(b)
            print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    sess.run(y, feed_dict={x: inputX })
    sess.run(tf.nn.softmax([1., 2.]))

    return 0


def main():

    classify_house_prices()

    return 0



if __name__ == '__main__':
    main()

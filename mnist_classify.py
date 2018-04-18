import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from datetime import timedelta
import math


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\ImageClassifier')


filter_size_one = 5          # Convolution filters are 5 x 5 pixels.
num_filters_one = 16         # There are 16 of these filters.
filter_size_two = 5          # Convolution filters are 5 x 5 pixels.
num_filters_two = 36         # There are 36 of these filters.

fully_conn_size = 128        # Number of neurons in fully-connected layer.
img_size = 28                # Image hight and width.
num_classes = 10             # Number of classes, one class for each of 10 digits.


def load_data():
    data = input_data.read_data_sets('./data/MNIST/', one_hot=True)
    return data


def plot_images(images, cls_true, cls_pred=None):
    # Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
    assert len(images) == len(cls_true) == 9
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def main():


    return 0


if __name__ == '__main__':
    main()


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import data_utils
import matplotlib.pyplot as plt


data_set_file = "./data_set.pkl"

language_one = data_utils.language_one
language_two = data_utils.language_two


def read_dataset(file_path):

    if os.path.exists(file_path):
        return data_utils.read_dataset(file_path)
    else:
        print("No file found: {0}".format(file_path))
        sys.exit(1)


def inspect_data(X, Y, en_idx2word, es_idx2word):
    print("Sentence in {0} - encoded: {1}".format(language_one, X[0]))
    print("Sentence in {0} - encoded: {1}".format(language_two, Y[0]))
    print("Decoded:\n------------------------")

    print_string = ""
    for i in range(len(X[1])):
        print_string = print_string + en_idx2word[X[1][i]] + " "
    print(print_string)
    print_string = ""
    for i in range(len(Y[1])):
        print_string = print_string + es_idx2word[Y[1][i]] + " "
    print(print_string)
    return 0


def data_padding(x, y, length = 15):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [en_word2idx['<pad>']]
        y[i] = [de_word2idx['<go>']] + y[i] + [de_word2idx['<eos>']] + (length-len(y[i])) * [de_word2idx['<pad>']]

def data_processing():

    return


def run_translator():
    X, Y, en_word2idx, en_idx2word, en_vocab, es_word2idx, es_idx2word, es_vocab = read_dataset(data_set_file)
    inspect_data(X, Y, en_idx2word, es_idx2word)

    return 0


def main():
    run_translator()
    return 0

if __name__ == '__main__':
    main()

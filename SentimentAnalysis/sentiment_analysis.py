from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
import os
import tflearn
from tflearn.datasets import imdb
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\SentimentAnalysis')

ign_dataset_path = "./ign_dataset.csv"
ign_model_path = "./ign_model.tfl"
imdb_dataset_path = "./imdb_dataset.pkl"
imdb_model_path = "./imdb_model.tfl"

save_model = True


def check_file_exist(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False


def get_categorical(array, nb_classes=11):
    '''
    Because tflearn to_categorical seemed to not work properly
    '''
    # array = total_Y[0]
    categorical_list = [0]*nb_classes
    value = array[0]
    categorical_list[value] = 1
    return np.array(categorical_list)


def run_on_imdb():
    # IMDB Dataset loading
    train, test, _ = imdb.load_data(path=imdb_dataset_path, n_words=10000, valid_portion=0.1)
    trainX, trainY = train
    testX, testY = test

    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)

    if check_file_exist(imdb_model_path):
        model.load(imdb_model_path)

    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,batch_size=32)

    if save_model:
        print("Saving model as 'imdb_model.tfl'")
        model.save(imdb_model_path)

    return 0



def run_on_ign():

    df_dataset = pd.read_csv(ign_dataset_path)
    df_dataset.set_index(['index'], inplace=True)
    # fill null values with empty strings
    df_dataset.fillna(value='', inplace=True)
    # extract the required columns for inputs and outputs
    data_X = df_dataset.title
    # data_X[0]
    label_Y = df_dataset.score_phrase
    # label_Y[5]

    # convert the strings in the input into integers corresponding to the dictionary positions
    # maps documents to sequences of word ids
    # data is automatically padded so we need to pad_sequences manually
    vocab_proc = VocabularyProcessor(15)
    total_X = np.array(list(vocab_proc.fit_transform(data_X)))
    # total_X[0]

    # we will have 11 classes in total for prediction, indices from 0 to 10
    # vocabulary processor for single word
    vocab_proc2 = VocabularyProcessor(1)
    total_Y = np.array(list(vocab_proc2.fit_transform(label_Y))) - 1
    # total_Y[5]
    # len(total_Y)

    # as we have 11 unique score_phrase
    # convert the indices into 11 dimensional vectors
    # This is wrong as it generate same array for different score_phrase
    # total_Y = to_categorical(total_Y, nb_classes=11)
    array_list = []
    for array in total_Y:
        array_list.append(get_categorical(array, 11))
    total_Y = np.array(array_list)
    # total_Y[4]

    # split into training and testing data
    train_X, test_X, train_Y, test_Y = train_test_split(total_X, total_Y, test_size=0.1)

    # build the network for classification
    # each input has length of 15
    net = tflearn.input_data([None, 15])

    # the 15 input word integers are then casted out into 256 dimensions each creating a word embedding.
    # we assume the dictionary has 10000 words maximum
    net = tflearn.embedding(net, input_dim=10000, output_dim=256)
    # each input would have a size of 15x256 and each of these 256 sized vectors are fed into the LSTM layer one at a time.
    # all the intermediate outputs are collected and then passed on to the second LSTM layer.
    net = tflearn.gru(net, 256, dropout=0.9, return_seq=True)
    # using the intermediate outputs, we pass them to another LSTM layer and collect the final output only this time
    net = tflearn.gru(net, 256, dropout=0.9)
    # the output is then sent to a fully connected layer that would give us our final 11 classes
    net = tflearn.fully_connected(net, 11, activation='softmax')
    # we use the adam optimizer instead of standard SGD since it converges much faster
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)

    if check_file_exist(ign_model_path):
        model.load(ign_model_path)

    model.fit(train_X, train_Y, validation_set=(test_X, test_Y), show_metric=True, batch_size=32, n_epoch=20)

    if save_model:
        print("Saving model as './ign_model.tfl'")
        model.save(ign_model_path)

    return 0


def main():

    run_on_imdb()
    # run_on_ign()

    return 0



if __name__ == '__main__':
    main()

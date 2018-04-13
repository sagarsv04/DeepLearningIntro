from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
import os
import tflearn
from tflearn.datasets import imdb
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor

# imports for GOT dataset
import codecs # encoding. word encodig
import glob # finds all pathnames matching a pattern, like regex
import logging # log events for libraries
import multiprocessing # concurrency
import pprint # pretty print, human readable
import re # regular expressions
import nltk # natural language toolkit
import gensim.models.word2vec as w2v # word 2 vec
import sklearn.manifold # dimensionality reduction
import matplotlib.pyplot as plt # plotting
import seaborn as sns # visualization


# os.getcwd()
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\SentimentAnalysis')


ign_dataset_path = "./ign_dataset.csv"
ign_model_path = "./ign_model.tfl"
imdb_dataset_path = "./imdb_dataset.pkl"
imdb_model_path = "./imdb_model.tfl"
got_data_base = "./got_data/"
got_model_path = "./thrones2vec.w2v"

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



def sentence_to_wordlist(raw):
    # convert into list of words
    # remove unecessary characters, split into words, no hyhens and shit
    # split into words
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


def plot_region(points, x_bounds, y_bounds):
    # points, x_bounds, y_bounds = points, (0, 1), (4, 4.5)

    sns.set_context("poster")
    slice = points[(x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) &
                    (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])]

    if len(slice):
        ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))

        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

        plt.show()
    else:
        print("Nothing to display in this region.")

    return 0


def nearest_similarity_cosmul(model, start1, end1, end2):
    similarities = model.most_similar_cosmul(positive=[end2, start1], negative=[end1])
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


def run_on_got():

    # get the book names, matching txt file
    book_filenames = sorted(glob.glob(got_data_base+"*.txt"))
    print("Found books:", book_filenames)

    # initialize rawunicode , we'll add all text to this one bigass file in memory
    corpus_raw = u""
    # corpus_raw = corpus_raw[:169]
    # for each book, open it, read it in utf-8 format
    # add it to the raw corpus
    for book_filename in book_filenames:
        # book_filename = book_filenames[0]
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))

    # tokenizastion! saved the trained model here
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # tokenize into sentences
    raw_sentences = tokenizer.tokenize(corpus_raw)

    # for each sentece, sentences where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        # raw_sentence = raw_sentences[0]
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))

    # count tokens, each one being a sentence
    token_count = sum([len(sentence) for sentence in sentences])
    print("The book corpus contains {0} tokens".format(token_count))

    #### define hyperparameters ####

    # 3 main tasks that vectors help with
    # DISTANCE, SIMILARITY, RANKING

    # dimensionality of the resulting word vectors.
    # more dimensions, more computationally expensive to train
    # but also more accurate
    # more dimensions = more generalized
    num_features = 300
    # minimum word count threshold.
    min_word_count = 3
    # number of threads to run in parallel.
    # more workers, faster we train
    num_workers = multiprocessing.cpu_count()
    # context window length.
    context_size = 7
    # downsample setting for frequent words.
    # 0 - 1e-5 is good for this
    downsampling = 1e-3
    # seed for the RNG, to make the results reproducible.
    # random number generator
    # deterministic, good for debugging
    seed = 1

    if check_file_exist(got_model_path):
        thrones2vec = w2v.Word2Vec.load(got_model_path)
    else:
        thrones2vec = w2v.Word2Vec(sg=1, seed=seed, workers=num_workers, size=num_features, min_count=min_word_count, window=context_size, sample=downsampling)
        thrones2vec.build_vocab(sentences)

    print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))

    thrones2vec.train(sentences, total_examples=thrones2vec.corpus_count, epochs=thrones2vec.epochs)

    if save_model:
        print("Saving model as './thrones2vec.w2v'")
        thrones2vec.save(got_model_path)

    # compress the word vectors into 2D space and plot them
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    all_word_vectors_matrix = thrones2vec.wv.syn0

    # train t-SNE, this could take a minute or two
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

    # plot the big picture
    # here we create a dataframe consisting list of words and there vector representation in 2D
    points = pd.DataFrame([(word, coords[0], coords[1])
                            for word, coords in [(word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
                                for word in thrones2vec.wv.vocab]],columns=["word", "x", "y"])

    sns.set_context("poster")
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    plt.show()

    # zoom in to some interesting places
    # people related to Kingsguard ended up together
    plot_region(points, x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))

    # food products are grouped nicely as well.
    plot_region(points, x_bounds=(0, 1), y_bounds=(4, 4.5))

    # words closest to the given word
    print("Word closest to Stark",thrones2vec.most_similar("Stark"))
    print("Word closest to Aerys",thrones2vec.most_similar("Aerys"))
    print("Word closest to direwolf",thrones2vec.most_similar("direwolf"))

    nearest_similarity_cosmul(thrones2vec, "Stark", "Winterfell", "Riverrun")
    nearest_similarity_cosmul(thrones2vec, "Jaime", "sword", "wine")
    nearest_similarity_cosmul(thrones2vec, "Arya", "Nymeria", "dragons")

    return 0


def main():

    # run_on_imdb()
    # run_on_ign()
    run_on_got()

    return 0


if __name__ == '__main__':
    main()

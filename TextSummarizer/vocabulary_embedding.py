
from collections import Counter
from itertools import chain
import os
import pickle
import matplotlib.pyplot as plt
from summerize import load_data_from_pickle, check_file_exist
import urllib.request
import shutil
import zipfile


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\TextSummarizer')


vocabulary_file_path = "./vocabulary_embedding.pickle"
pickle_file_path = "./bbc/bbc_data.pickle"
data_dir_base = "./bbc/"

seed = 42
vocab_size = 40000
embedding_dim = 100
to_lower = False  # dont lower case the text

glove_file_name = 'glove_6B_%dd.txt'%embedding_dim

empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word


def get_vocab_count(pickle_file_path):
	# read pickle file and create vocab_count
	vocab_count = Counter()
	if check_file_exist(pickle_file_path):
		with open(pickle_file_path, 'rb') as f:
			for title, body in load_data_from_pickle(f):
				vocab_count.update(word for word in title.split())
				vocab_count.update(word for word in body.split())

		print("Vocab count:", len(vocab_count))
		return vocab_count
	else:
		print("Pickle file: {0} not found.".format(pickle_file_path))
		return -1



def get_vocab_from_vocab_count(vocab_count):
	# get list of all the words
	vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))
	return vocab


def plot_word_distribution(vocab_count, vocab):

	plt.plot([vocab_count[word] for word in vocab]);
	plt.gca().set_xscale("log", nonposx='clip')
	plt.gca().set_yscale("log", nonposy='clip')
	plt.title('Word distribution in headlines and discription')
	plt.xlabel('Rank')
	plt.ylabel('Total appearances');
	plt.show()

	return 0


def get_idx(vocab):
	# indexing words
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
	# check if below is correct
    idx2word = dict((idx,word) for word,idx in word2idx.items())
    return word2idx, idx2word


def word_embedding(glove_file_name):

    glove_data_dir = os.path.join(data_dir_base, 'glove_data/')
	glove_file_path = os.path.join(glove_data_dir, glove_file_name)
	if not check_file_exist(glove_file_path):
		download_path = data_dir_base+'glove_6B.zip'
		print("Downloading file: {0}.".format(download_path))
		# Download the file from `url` and save it locally under `file_name`:
		with urllib.request.urlopen("http://nlp.stanford.edu/data/glove.6B.zip") as response, open(download_path, 'wb') as out_file:
    		shutil.copyfileobj(response, out_file)



	return 0



def main():

	vocab_count = get_vocab_count(pickle_file_path)
	if not(vocab_count==-1):
		vocab = get_vocab_from_vocab_count(vocab_count)
		plot_word_distribution(vocab_count, vocab)
		word2idx, idx2word = get_idx(vocab)


	return 0


if __name__ == '__main__':
	main()

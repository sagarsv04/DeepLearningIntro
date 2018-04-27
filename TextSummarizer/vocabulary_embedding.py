
from collections import Counter
from itertools import chain
from summerize import load_data_from_pickle
import os
import pickle
import matplotlib.pyplot as plt


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\TextSummarizer')


vocabulary_file_path = "./vocabulary_embedding.pickle"
pickle_file_path = "./bbc/bbc_data.pickle"

seed = 42
vocab_size = 40000
embedding_dim = 100
to_lower = False  # dont lower case the text

empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word


def get_vocab_count():
	# read pickle file and create vocab_count
	vocab_count = Counter()
	with open(pickle_file_path, 'rb') as f:
		for title, body in load_data_from_pickle(f):
			vocab_count.update(word for word in title.split())
			vocab_count.update(word for word in body.split())
	print("Vocab count:", len(vocab_count))
	return vocab_count


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
	# Index words
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
	# check if below is correct
    idx2word = dict((idx,word) for word,idx in enumerate(word2idx))
	word2idx.
    return word2idx, idx2word


def main():

	vocab_count = get_vocab_count()
	vocab = get_vocab_from_vocab_count(vocab_count)
	plot_word_distribution(vocab_count, vocab)

	return 0


if __name__ == '__main__':
	main()

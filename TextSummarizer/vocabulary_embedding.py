
from collections import Counter
from itertools import chain
import os
import pickle
import matplotlib.pyplot as plt
from summerize import load_data_from_pickle, check_file_exist
import urllib.request
import shutil
import zipfile
import numpy as np


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\TextSummarizer')


vocabulary_file_path = "./vocabulary_embedding.pickle"
data_set_file_path = "./data_set.pickle"
pickle_file_path = "./bbc/bbc_data.pickle"
data_dir_base = "./bbc/"

seed = 42
vocab_size = 40000
embedding_dim = 100
to_lower = False  # dont lower case the text
glove_match_threshold = 0.5

glove_dir_name = 'glove_6B'
glove_file_name = 'glove.6B.%dd.txt'%embedding_dim
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


def download_glove_corpus(glove_data_dir):

	download_path = data_dir_base+'glove_6B.zip'
	if not check_file_exist(download_path):
		print("Downloading file: {0}.".format(download_path))
		# Download the file from `url` and save it locally under `file_name`:
		with urllib.request.urlopen("http://nlp.stanford.edu/data/glove.6B.zip") as response, open(download_path, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)
	print("Extracting file: {0}.".format(download_path))
	with zipfile.ZipFile(download_path,"r") as zip_ref:
		zip_ref.extractall(glove_data_dir)

	return 0


def get_glove_embedding_weights(glove_num_of_symbols, glove_file_path):

	glove_index_dict = {}
	glove_embedding_weights = np.empty((glove_num_of_symbols, embedding_dim))
	globale_scale = 0.1
	# fb = open(glove_file_path, 'r', encoding="utf8")
	# index, line = 0,fb.readlines(1)[0]
	# fb.close()
	with open(glove_file_path, 'r', encoding="utf8") as fp:
		for index, line in enumerate(fp):
			line = line.strip().split()
			word = line[0]
			glove_index_dict[word] = index
			glove_embedding_weights[index,:] = list(map(float,line[1:]))
	glove_embedding_weights *= globale_scale
	print("Std. of glove_embedding_weights: {0}".format(glove_embedding_weights.std()))
	# check keys with case sensitive
	# add word in lower case with index
	for word, index in glove_index_dict.items():
		word = word.lower()
		if word not in glove_index_dict:
			glove_index_dict[word] = index

	return glove_embedding_weights, glove_index_dict


def get_our_embedding_matrix(glove_embedding_weights, glove_index_dict, idx2word):
	# use GloVe to initialize embedding matrix
	# generate random embedding with same scale as glove
	np.random.seed(seed)
	shape = (vocab_size, embedding_dim)
	scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
	embedding_matrix = np.random.uniform(low=-scale, high=scale, size=shape)
	print("random_embedding_std/glove_scale: {0}/{1}".format(embedding_matrix.std(), scale))
	# copy from glove weights of words that appear in our short vocabulary (idx2word)
	count = 0
	for i in range(vocab_size):
		# i = 2
		word = idx2word[i]
		glove_index = glove_index_dict.get(word, glove_index_dict.get(word.lower()))
		if glove_index is None and word.startswith('#'): # glove has no hastags (I think...)
			word = word[1:]
			glove_index = glove_index_dict.get(word, glove_index_dict.get(word.lower()))
		if glove_index is not None:
			embedding_matrix[i,:] = glove_embedding_weights[glove_index,:]
			count +=1
	print("Number of tokens, in small vocab, found in glove and copied to embedding: {0}, {1}".format(count, count/float(vocab_size)))
	return embedding_matrix


def get_our_word2glove(glove_index_dict, word2idx):

	# lots of word in the full vocabulary (word2idx) are outside vocab_size.
	# Build an alterantive which will map them to their closest match in glove
	# but only if the match is good enough (cos distance above glove_match_threshold)

	word2glove = {}
	for word in word2idx:
		# word = "copy"
		if word in glove_index_dict:
			glove_word = word
		elif word.lower() in glove_index_dict:
			glove_word = word.lower()
		elif word.startswith('#') and word[1:] in glove_index_dict:
			glove_word = word[1:]
		elif word.startswith('#') and word[1:].lower() in glove_index_dict:
			glove_word = word[1:].lower()
		else:
			continue
		word2glove[word] = glove_word
	return word2glove


def get_glove_match(glove_index_dict, glove_embedding_weights, embedding_matrix, word2idx, idx2word, word2glove):

	normed_embedding = embedding_matrix/np.array([np.sqrt(np.dot(glove_weight,glove_weight)) for glove_weight in embedding_matrix])[:,None]
	nb_unknown_words = 100
	glove_match = []
	for word, index in word2idx.items():
		# word, index = "copy", word2idx["copy"]
		if index >= vocab_size - nb_unknown_words and word.isalpha() and word in word2glove:
			glove_index = glove_index_dict[word2glove[word]]
			glove_weight = glove_embedding_weights[glove_index,:].copy()
			# find row in embedding that has the highest cos score with glove_weight
			glove_weight /= np.sqrt(np.dot(glove_weight,glove_weight))
			score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], glove_weight)
			while True:
				embedding_index = score.argmax()
				s = score[embedding_index]
				if s < glove_match_threshold:
					break
				if idx2word[embedding_index] in word2glove:
					glove_match.append((word, embedding_index, s))
					break
				score[embedding_index] = -1
	glove_match.sort(key = lambda x: -x[2])
	print("Number of glove substitutes found: {0}".format(len(glove_match)))
	return glove_match


def print_worst_substitutions(glove_match, idx2word):
	for orig, sub, score in glove_match[-10:]:
		print("Score:{0}, word:{1}, subs:{2}".format(score, orig, idx2word[sub]))
	return 0


def plot_x_y_data(word2idx, pickle_file_path):

	title_lenght = []
	body_lenght = []
	with open(pickle_file_path, 'rb') as f:
		for title, body in load_data_from_pickle(f):
			y = [word2idx[token] for token in title.split()]
			x = [word2idx[token] for token in body.split()]
			title_lenght.append(len(y))
			body_lenght.append(len(x))
			# if to_lower:
			# 	head_list.append(title.lower())
			# else:
			# 	head_list.append(title)
	fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
	fig.suptitle('Title and body words count')
	axs[0].hist(title_lenght, bins=50)
	axs[1].hist(body_lenght, bins=50)
	plt.show()

	return 0


def save_data_to_pickle(embedding_matrix, idx2word, word2idx, glove_index2index, pickle_file_path):

	if not check_file_exist(vocabulary_file_path):
		print("Storing vocabulary data to: {0}".format(vocabulary_file_path))
		with open(vocabulary_file_path,'wb') as fp:
			pickle.dump((embedding_matrix, idx2word, word2idx, glove_index2index), fp)
	else:
		print("File already exist: {0}".format(vocabulary_file_path))

	if not check_file_exist(data_set_file_path):
		print("Storing data set to: {0}".format(data_set_file_path))
		data_set_length = 0
		with open(pickle_file_path, 'rb') as f:
			for title, body in load_data_from_pickle(f):
				with open(data_set_file_path, 'ab') as fb:
					if to_lower:
						title = title.lower()
						body = body.lower()
					# X,Y = title, body
					pickle.dump((title, body), fb)
					data_set_length += 1
		if data_set_length > 0:
			file_ext = data_set_file_path.split('.')[-1]
			with open(data_set_file_path[:-len('.'+file_ext)]+'_size.'+file_ext, 'wb') as f:
				pickle.dump({"size": data_set_length}, f)
	else:
		print("File already exist: {0}".format(data_set_file_path))

	return 0


def save_word_embedding_data(glove_dir_name, word2idx, idx2word, pickle_file_path):

	glove_data_dir = os.path.join(data_dir_base, glove_dir_name)
	if not os.path.exists(glove_data_dir):
		download_glove_corpus(glove_data_dir)
	# count number of lines in files ie: number of symbols
	glove_file_path = glove_data_dir+'/'+glove_file_name
	glove_num_of_symbols = sum(1 for line in open(glove_file_path, encoding="utf8"))

	glove_embedding_weights, glove_index_dict = get_glove_embedding_weights(glove_num_of_symbols, glove_file_path)
	embedding_matrix = get_our_embedding_matrix(glove_embedding_weights, glove_index_dict, idx2word)
	word2glove = get_our_word2glove(glove_index_dict, word2idx)
	glove_match = get_glove_match(glove_index_dict, glove_embedding_weights, embedding_matrix, word2idx, idx2word, word2glove)
	print_worst_substitutions(glove_match, idx2word)
	# build a lookup table of index of outside words to index of inside words
	glove_index2index = dict((word2idx[word],embedding_index) for word, embedding_index, _ in glove_match)
	# work on this to make it look better
	plot_x_y_data(word2idx, pickle_file_path)

	save_data_to_pickle(embedding_matrix, idx2word, word2idx, glove_index2index, pickle_file_path)

	return 0


def main():

	vocab_count = get_vocab_count(pickle_file_path)
	if not(vocab_count==-1):
		vocab = get_vocab_from_vocab_count(vocab_count)
		plot_word_distribution(vocab_count, vocab)
		word2idx, idx2word = get_idx(vocab)
		save_word_embedding_data(glove_dir_name, word2idx, idx2word, pickle_file_path)
	return 0


if __name__ == '__main__':
	main()

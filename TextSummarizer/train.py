import pickle
import os
from sklearn.cross_validation import train_test_split

# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\TextSummarizer')


vocabulary_file_path = "./vocabulary_embedding.pickle"
data_set_file_path = "./data_set.pickle"


maxlend = 25 # 0 - if we dont want to use description at all
maxlenh = 25
maxlen = maxlend + maxlenh
num_unknown_words = 10
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm = False

# the out of the first activation_rnn_size nodes from the top LSTM layer will be used for activation and the rest will be used to select predicted word
activation_rnn_size = 40 if maxlend else 0

# training parameters
seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size = 64
nflips = 10

num_train_samples = 30000
num_val_samples = 3000


def get_word_embedding_data(vocabulary_file_path):

	with open(vocabulary_file_path, 'rb') as fp:
		embedding_matrix, idx2word, word2idx, glove_index2index = pickle.load(fp)

	vocab_size, embedding_size = embedding_matrix.shape
	# print() 'number of examples',len(X),len(Y)
	print("Dimension of embedding space for words: {0}".format(embedding_size))
	print("Vocabulary size: {0}, the last {1} words can be used as place holders for unknown/oov words.".format(vocab_size, num_unknown_words))
	print("Total number of different words", len(idx2word), len(word2idx))
	print("Number of words outside vocabulary which we can substitue using glove similarity", len(glove_index2index))
	print("Number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)", len(idx2word)-vocab_size-len(glove_index2index))

	for i in range(nb_unknown_words):
    	idx2word[vocab_size-1-i] = '<%d>'%i

	oov = vocab_size-nb_unknown_words
	for i in range(oov, len(idx2word)):
    	idx2word[i] = idx2word[i]+'^'

	return embedding_matrix, idx2word, word2idx, glove_index2index


def get_data_set(data_set_file_path):
	# read data from pickle and create dataset

	return X_train, X_test, Y_train, Y_test


def run_traing_processe(vocabulary_file_path, data_set_file_path):

	embedding_matrix, idx2word, word2idx, glove_index2index = get_word_embedding_data(vocabulary_file_path)
	X_train, X_test, Y_train, Y_test = get_data_set(data_set_file_path)


	return 0

def main():

	run_traing_processe(vocabulary_file_path, data_set_file_path)

	return 0


if __name__ == '__main__':
	main()

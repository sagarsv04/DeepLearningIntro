import pickle
import os
from summerize import load_data_from_pickle, check_file_exist
import random, sys
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras.layers.core import Lambda
from keras.utils import np_utils
import keras.backend as K
from keras.optimizers import Adam, RMSprop # usually I prefer Adam but article used rmsprop
# opt = Adam(lr=LR)  # keep calm and reduce learning rate


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')
# os.chdir(r'D:\CodeRepo\DeepLearningIntro\TextSummarizer')


vocabulary_file_path = "./vocabulary_embedding.pickle"
data_set_file_path = "./data_set.pickle"

read_percentage = 100 # percent of data to read from stored pickle data_set
percentage_test_samples = 20 # percent of data as test from X, Y data_Set
percentage_train_samples = 100-percentage_test_samples

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
empty = 0
eos = 1
seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
dropout = 0
recurrent_dropout = 0
optimizer = 'adam'
LR = 1e-4
batch_size = 64
nflips = 10


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

	for i in range(num_unknown_words):
		idx2word[vocab_size-1-i] = '<%d>'%i

	oov = vocab_size-num_unknown_words
	for i in range(oov, len(idx2word)):
		idx2word[i] = idx2word[i]+'^'

	return embedding_matrix, idx2word, word2idx, glove_index2index, vocab_size, embedding_size, oov


def get_x_y_data_set(data_set_file_path):
	# read data from pickle and create dataset
	X = []
	Y = []
	iterate_upto = 0
	file_ext = data_set_file_path.split('.')[-1]
	data_size_file = data_set_file_path[:-len('.'+file_ext)]+'_size.'+file_ext
	if check_file_exist(data_size_file):
		with open(data_size_file, 'rb') as f:
			for size_data in load_data_from_pickle(f):
				iterate_upto = int(size_data['size']*(min(read_percentage, 100)/100))

		if check_file_exist(data_set_file_path):
			if iterate_upto > 0:
				with open(data_set_file_path, 'rb') as f:
					counter = 0
					for title, body in load_data_from_pickle(f):
						counter += 1
						X.append(title)
						Y.append(body)
						if counter >= iterate_upto:
							break
				print("Number of data examples, X: {0}, Y: {1}".format(len(X), len(Y)))
			else:
				print("Iteration: {0} is to small".format(iterate_upto))
		else:
			print("Pickle file: {0} not found.".format(data_set_file_path))
	else:
		print("Pickle file: {0} not found.".format(data_size_file))

	return	X, Y


def get_train_test_data(data_set_file_path):

	X, Y = get_x_y_data_set(data_set_file_path)
	X_train, X_test, Y_train, Y_test = [], [], [], []
	if len(X) > 0 and len(Y) > 0:
		test_size = int(len(X)*(percentage_test_samples/100))
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	return X_train, X_test, Y_train, Y_test


def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
	desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]

	# RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
	# activation for every head word and every desc word
	activation_energies = K.batch_dot(head_activations, desc_activations, axes=[2,2])
	# make sure we dont use description words that are masked out
	activation_energies = activation_energies + -1e20*K.expand_dims(-K.cast(mask[:, :maxlend],dtype='float32'),axis=1)

	# for every head word compute weights for every desc word
	activation_energies = K.reshape(activation_energies,(-1,maxlend))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

	# for every head word compute weighted average of desc words
	desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=[2,1])
	return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
	def __init__(self,**kwargs):
		super(SimpleContext, self).__init__(simple_context,**kwargs)
		self.supports_masking = True

	def compute_mask(self, input, input_mask=None):
		return input_mask[:, maxlend:]

	def get_output_shape_for(self, input_shape):
		nb_samples = input_shape[0]
		n = 2*(rnn_size - activation_rnn_size)
		return (nb_samples, maxlenh, n)


def create_model(embedding_matrix, vocab_size, embedding_size):

	regularizer = l2(weight_decay) if weight_decay else None
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size,
						name='embedding_1',
						mask_zero=True,
						embeddings_regularizer=regularizer,
						weights=[embedding_matrix],
						input_length=maxlen))
	for i in range(rnn_layers):
		lstm = LSTM(rnn_size,
					name='lstm_%d'%(i+1),
					recurrent_regularizer=regularizer,
					kernel_regularizer=regularizer,
					dropout=dropout,
					return_sequences=True,
					recurrent_dropout=recurrent_dropout,
					bias_regularizer=regularizer)
		model.add(lstm)
		model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))

	if activation_rnn_size:
		model.add(SimpleContext(name='simplecontext_1'))
		model.add(TimeDistributed(Dense(vocab_size, name="timedistributed_1", kernel_regularizer=regularizer, bias_regularizer=regularizer)))
		model.add(Activation('softmax', name='activation_1'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	K.set_value(model.optimizer.lr,np.float32(LR))

	return model


def flip_headline(x, oov, nflips=None, model=None, debug=False):
	"""given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
	with words predicted by the model
	"""
	if nflips is None or model is None or nflips <= 0:
		return x

	batch_size = len(x)
	assert np.all(x[:,maxlend] == eos)
	probs = model.predict(x, verbose=0, batch_size=batch_size)
	x_out = x.copy()
	for b in range(batch_size):
		# pick locations we want to flip
		# 0...maxlend-1 are descriptions and should be fixed
		# maxlend is eos and should be fixed
		flips = sorted(random.sample(xrange(maxlend+1,maxlen), nflips))
		if debug and b < debug:
			print("b:", b)
		for input_idx in flips:
			if x[b,input_idx] == empty or x[b,input_idx] == eos:
				continue
			# convert from input location to label location
			# the output at maxlend (when input is eos) is feed as input at maxlend+1
			label_idx = input_idx - (maxlend+1)
			prob = probs[b, label_idx]
			w = prob.argmax()
			if w == empty:  # replace accidental empty with oov
				w = oov
			if debug and b < debug:
				print("{0} => {1}".format(idx2word[x_out[b,input_idx]], idx2word[w])))
			x_out[b,input_idx] = w
		if debug and b < debug:
			print("")
	return x_out


def conv_seq_labels(xds, xhs, oov, vocab_size, nflips=None, model=None, debug=False):
	"""description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
	batch_size = len(xhs)
	assert len(xds) == batch_size
	x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
	x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
	x = flip_headline(x, oov, nflips=nflips, model=model, debug=debug)

	y = np.zeros((batch_size, maxlenh, vocab_size))
	for i, xh in enumerate(xhs):
		xh = vocab_fold(xh) + [eos] + [empty]*maxlenh  # output does have a eos at end
		xh = xh[:maxlenh]
		y[i,:,:] = np_utils.to_categorical(xh, vocab_size)

	return x, y


def gen(Xd, Xh, oov, vocab_size, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    """yield batches. for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches

    while training it is good idea to flip once in a while the values of the headlines from the
    value taken from Xh to value generated by the model.
    """
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxint)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, oov, vocab_size, nflips=nflips, model=model, debug=debug)


def run_traing_processe(vocabulary_file_path, data_set_file_path):

	# seed weight initialization
	random.seed(seed)
	np.random.seed(seed)
	embedding_matrix, idx2word, word2idx, glove_index2index, vocab_size, embedding_size, oov = get_word_embedding_data(vocabulary_file_path)
	X_train, X_test, Y_train, Y_test = get_train_test_data(data_set_file_path)
	model = create_model(embedding_matrix, vocab_size, embedding_size)
	if len(X_train) > 0:



	return 0

def main():

	run_traing_processe(vocabulary_file_path, data_set_file_path)

	return 0


if __name__ == '__main__':
	main()

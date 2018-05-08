
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

# ops and hyperparameters# ops a
learning_rate = 5e-3
batch_size = 64
steps = 500
input_seq_len = 15
output_seq_len = 17

# let's translate these sentences
en_sentences = ["What' s your name", 'My name is', 'What are you doing', 'I am reading a book',\
				'How are you', 'I am good', 'Do you speak English', 'What time is it', 'Hi', 'Goodbye', 'Yes', 'No']

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


def data_padding(X, Y, en_word2idx, es_word2idx, length = 15):
	for i in range(len(X)):
		# i = 0
		X[i] = X[i] + (length - len(X[i])) * [en_word2idx['<pad>']]
		Y[i] = [es_word2idx['<go>']] + Y[i] + [es_word2idx['<eos>']] + (length-len(Y[i])) * [es_word2idx['<pad>']]
	return X, Y

def data_processing(X, Y, en_word2idx, es_word2idx):
	X, Y = data_padding(X, Y, en_word2idx, es_word2idx)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)
	return X_train, X_test, Y_train, Y_test

def softmax(x):
	n = np.max(x)
	e_x = np.exp(x - n)
	return e_x / e_x.sum()

# feed data into placeholders
def feed_dict(x, y, targets, es_word2idx, encoder_inputs, decoder_inputs, target_weights, batch_size = 64):
	feed = {}
	idxes = np.random.choice(len(x), size = batch_size, replace = False)
	for i in range(input_seq_len):
		feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes], dtype = np.int32)

	for i in range(output_seq_len):
		feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes], dtype = np.int32)

	feed[targets[len(targets)-1].name] = np.full(shape = [batch_size], fill_value = es_word2idx['<pad>'], dtype = np.int32)

	for i in range(output_seq_len-1):
		batch_weights = np.ones(batch_size, dtype = np.float32)
		target = feed[decoder_inputs[i+1].name]
		for j in range(batch_size):
			if target[j] == es_word2idx['<pad>']:
				batch_weights[j] = 0.0
		feed[target_weights[i].name] = batch_weights

	feed[target_weights[output_seq_len-1].name] = np.zeros(batch_size, dtype = np.float32)

	return feed

# forward step
def forward_step(sess, feed, outputs_proj):
	output_sequences = sess.run(outputs_proj, feed_dict = feed)
	return output_sequences

# training step
def backward_step(sess, feed, optimizer):
	sess.run(optimizer, feed_dict = feed)

# decode output sequence
def decode_output(output_seq, es_idx2word):
	words = []
	for i in range(output_seq_len):
		smax = softmax(output_seq[i])
		idx = np.argmax(smax)
		words.append(es_idx2word[idx])
	return words

def run_translator():
	X, Y, en_word2idx, en_idx2word, en_vocab, es_word2idx, es_idx2word, es_vocab = read_dataset(data_set_file)
	inspect_data(X, Y, en_idx2word, es_idx2word)
	X_train, X_test, Y_train, Y_test = data_processing(X, Y, en_word2idx, es_word2idx)

	# build a model
	en_vocab_size = len(en_vocab) + 2 # + <pad>, <ukn>
	es_vocab_size = len(es_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>

	def sampled_loss(labels, logits):
		return tf.nn.sampled_softmax_loss(weights = w_t, biases = b, labels = tf.reshape(labels, [-1, 1]), inputs = logits, num_sampled = 512, num_classes = es_vocab_size)

	# placeholders
	encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
	decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

	targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]
	# add one more target
	targets.append(tf.placeholder(dtype = tf.int32, shape = [None], name = 'last_target'))
	target_weights = [tf.placeholder(dtype = tf.float32, shape = [None], name = 'target_w{}'.format(i)) for i in range(output_seq_len)]

	# output projection
	size = 512
	w_t = tf.get_variable('proj_w', [es_vocab_size, size], tf.float32)
	b = tf.get_variable('proj_b', [es_vocab_size], tf.float32)
	w = tf.transpose(w_t)
	output_projection = (w, b)

	outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,
																			decoder_inputs,
																			tf.contrib.rnn.BasicLSTMCell(size),
																			num_encoder_symbols = en_vocab_size,
																			num_decoder_symbols = es_vocab_size,
																			embedding_size = 100,
																			feed_previous = False,
																			output_projection = output_projection,
																			dtype = tf.float32)

	# define our loss function
	# sampled softmax loss - returns: A batch_size 1-D tensor of per-example sampled softmax losses
	# Weighted cross-entropy loss for a sequence of logits
	loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, targets, target_weights, softmax_loss_function = sampled_loss)
	# ops for projecting outputs
	outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]
	# training op
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
	# init op
	init = tf.global_variables_initializer()
	# let's train the model
	# we will use this list to plot losses through steps
	losses = []
	# save a checkpoint so we can restore the model later
	saver = tf.train.Saver()
	print('------------------TRAINING------------------')
	with tf.Session() as sess:
		sess.run(init)
		t = time.time()
		for step in range(steps):
			feed = feed_dict(X_train, Y_train, targets, es_word2idx, encoder_inputs, decoder_inputs, target_weights)
			backward_step(sess, feed, optimizer)
			if step % 5 == 4 or step == 0:
				loss_value = sess.run(loss, feed_dict = feed)
				print('step: {}, loss: {}'.format(step, loss_value))
				losses.append(loss_value)
			if step % 20 == 19:
				saver.save(sess, 'checkpoints/', global_step=step)
				print('Checkpoint is saved')

		print('Training time for {} steps: {}s'.format(steps, time.time() - t))

	with plt.style.context('fivethirtyeight'):
		plt.plot(losses, linewidth = 1)
		plt.xlabel('Steps')
		plt.ylabel('Losses')
		plt.ylim((0, 12))
	plt.show()

	# let's test the model
	with tf.Graph().as_default():

		# placeholders
		encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
		decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

		# output projection
		size = 512
		w_t = tf.get_variable('proj_w', [es_vocab_size, size], tf.float32)
		b = tf.get_variable('proj_b', [es_vocab_size], tf.float32)
		w = tf.transpose(w_t)
		output_projection = (w, b)

		# change the model so that output at time t can be fed as input at time t+1
		outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,
																				decoder_inputs,
																				tf.contrib.rnn.BasicLSTMCell(size),
																				num_encoder_symbols = en_vocab_size,
																				num_decoder_symbols = es_vocab_size,
																				embedding_size = 100,
																				feed_previous = True, # <-----this is changed----->
																				output_projection = output_projection,
																				dtype = tf.float32)

		# ops for projecting outputs
		outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

		# let's translate these sentences
		en_sentences_encoded = [[en_word2idx.get(word, 0) for word in en_sentence.split()] for en_sentence in en_sentences]

		# padding to fit encoder input
		for i in range(len(en_sentences_encoded)):
			en_sentences_encoded[i] += (15 - len(en_sentences_encoded[i])) * [en_word2idx['<pad>']]

		# restore all variables - use the last checkpoint saved
		saver = tf.train.Saver()
		path = tf.train.latest_checkpoint('checkpoints')

		with tf.Session() as sess:
			# restore
			saver.restore(sess, path)

			# feed data into placeholders
			feed = {}
			for i in range(input_seq_len):
				feed[encoder_inputs[i].name] = np.array([en_sentences_encoded[j][i] for j in range(len(en_sentences_encoded))], dtype = np.int32)

			feed[decoder_inputs[0].name] = np.array([es_word2idx['<go>']] * len(en_sentences_encoded), dtype = np.int32)

			# translate
			output_sequences = sess.run(outputs_proj, feed_dict = feed)

			# decode seq.
			for i in range(len(en_sentences_encoded)):
				print('{}.\n--------------------------------'.format(i+1))
				ouput_seq = [output_sequences[j][i] for j in range(output_seq_len)]
				#decode output sequence
				words = decode_output(ouput_seq, es_idx2word)

				print(en_sentences[i])
				print_string = ""
				for i in range(len(words)):
					if words[i] not in ['<eos>', '<pad>', '<go>']:
						print_string = print_string + words[i] + " "
				print(print_string)
				print('\n--------------------------------')

	return 0


def main():
	run_translator()
	return 0

if __name__ == '__main__':
	main()

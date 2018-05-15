import numpy as np # matrix math
import tensorflow as tf # machine learning
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import matplotlib.pyplot as plt

PAD = 0
EOS = 1

batch_sizes = 100
max_batches = 3001
batches_in_epoch = 1000

vocab_size = 10
input_embedding_size = 20 #character length

encoder_hidden_units = 20 #num neurons
decoder_hidden_units = encoder_hidden_units * 2


def helpers_batch(inputs, max_sequence_length=None):
	"""
	Args:
		inputs:
			list of sentences (integer lists)
		max_sequence_length:
			integer specifying how large should `max_time` dimension be.
			If None, maximum sequence length would be used

	Outputs:
		inputs_time_major:
			input sentences transformed into time-major matrix
			(shape [max_time, batch_size]) padded with 0s
		sequence_lengths:
			batch-sized list of integers specifying amount of active
			time steps in each input sequence
	"""

	sequence_lengths = [len(seq) for seq in inputs]
	batch_sizes = len(inputs)

	if max_sequence_length is None:
		max_sequence_length = max(sequence_lengths)

	inputs_batch_major = np.zeros(shape=[batch_sizes, max_sequence_length], dtype=np.int32) # == PAD

	for i, seq in enumerate(inputs):
		for j, element in enumerate(seq):
			inputs_batch_major[i, j] = element

	# [batch_size, max_time] -> [max_time, batch_size]
	inputs_time_major = inputs_batch_major.swapaxes(0, 1)

	return inputs_time_major, sequence_lengths


def helpers_random_sequences(length_from, length_to,
					 vocab_lower, vocab_upper,
					 batch_sizes):
	""" Generates batches of random integer sequences,
		sequence length in [length_from, length_to],
		vocabulary in [vocab_lower, vocab_upper]
	"""
	if length_from > length_to:
			raise ValueError('length_from > length_to')

	def random_length():
		if length_from == length_to:
			return length_from
		return np.random.randint(length_from, length_to + 1)

	while True:
		yield [np.random.randint(low=vocab_lower,
							  high=vocab_upper,
							  size=random_length()).tolist()
							  for _ in range(batch_sizes)]


def loop_fn_initial(decoder_lengths, eos_step_embedded, encoder_final_state):
	# manually specifying loop function through time - to get initial cell state and input to RNN
	# normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn
	# we define and return these values, no operations occur here
	initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
	# end of sentence
	initial_input = eos_step_embedded
	# last time steps cell state
	initial_cell_state = encoder_final_state
	# none
	initial_cell_output = None
	# none
	initial_loop_state = None  # we don't need to pass any additional information
	return (initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state)


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state, W, b, embeddings, decoder_lengths, pad_step_embedded):
	# attention mechanism --choose which previously generated token to pass as input in the next timestep
	def get_next_input():
		# dot product between previous ouput and weights, then + biases
		output_logits = tf.add(tf.matmul(previous_output, W), b)
		# Logits simply means that the function operates on the unscaled output of
		# earlier layers and that the relative scale to understand the units is linear.
		# It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities
		# (you might have an input of 5).
		# prediction value at current time step
		# Returns the index with the largest value across axes of a tensor.
		prediction = tf.argmax(output_logits, axis=1)
		# embed prediction for the next input
		next_input = tf.nn.embedding_lookup(embeddings, prediction)
		return next_input

	elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
												  # defining if corresponding sequence has ended
	# Computes the "logical and" of elements across dimensions of a tensor.
	finished = tf.reduce_all(elements_finished) # -> boolean scalar
	# Return either fn1() or fn2() based on the boolean predicate pred.
	input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
	# set previous to current
	state = previous_state
	output = previous_output
	loop_state = None

	return (elements_finished, input, state, output, loop_state)


def next_feed(batches, encoder_inputs, encoder_inputs_length, decoder_targets):
	batch = next(batches)
	encoder_inputs_, encoder_input_lengths_ = helpers_batch(batch)
	decoder_targets_, _ = helpers_batch([(sequence) + [EOS] + [PAD] * 2 for sequence in batch])
	return {encoder_inputs: encoder_inputs_,encoder_inputs_length: encoder_input_lengths_,decoder_targets: decoder_targets_}


def run_seq2seq():

	def loop_fn(time, previous_output, previous_state, previous_loop_state):
		if previous_state is None: # time == 0
			assert previous_output is None and previous_state is None
			return loop_fn_initial(decoder_lengths, eos_step_embedded, encoder_final_state)
		else:
			return loop_fn_transition(time, previous_output, previous_state, previous_loop_state, W, b, embeddings, decoder_lengths, pad_step_embedded)

	tf.reset_default_graph() # Clears the default graph stack and resets the global default graph.
	sess = tf.InteractiveSession() # initializes a tensorflow session
	# input placehodlers
	encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
	# contains the lengths for each of the sequence in the batch, we will pad so all the same
	# if you don't want to pad, check out dynamic memory networks to input variable length sequences
	encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
	decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
	# randomly initialized embedding matrrix that can fit input sequence
	# used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
	# reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
	embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
	# this thing could get huge in a real world application
	encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
	# Encoder
	encoder_cell = LSTMCell(encoder_hidden_units)
	((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
	(tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
									cell_bw=encoder_cell,
									inputs=encoder_inputs_embedded,
									sequence_length=encoder_inputs_length,
									dtype=tf.float32, time_major=True))
	# Concatenates tensors along one dimension.
	encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
	# letters h and c are commonly used to denote "output value" and "cell state".
	# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
	# Those tensors represent combined internal state of the cell, and should be passed together.
	encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
	encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
	# TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
	encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
	# Decoder
	decoder_cell = LSTMCell(decoder_hidden_units)
	# we could print this, won't need
	encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
	decoder_lengths = encoder_inputs_length + 3
	# +2 additional steps, +1 leading <EOS> token for decoder inputs
	# output(t) -> output projection(t) -> prediction(t) (argmax) -> input embedding(t+1) -> input(t+1)
	# manually specifying since we are going to implement attention details for the decoder in a sec
	# weights
	W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
	# bias
	b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
	# create padded inputs for the decoder from the word embeddings
	# were telling the program to test a condition, and trigger an error if the condition is false.
	assert EOS == 1 and PAD == 0
	eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
	pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')
	# retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
	eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
	pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
	# Creates an RNN specified by RNNCell cell and loop function loop_fn.
	# This function is a more primitive version of dynamic_rnn that provides more direct access to the
	# inputs each iteration. It also provides more control over when to start and finish reading the sequence,
	# and what to emit for the output.

	decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
	decoder_outputs = decoder_outputs_ta.stack()
	# Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
	# reduces dimensionality
	decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
	# flettened output tensor
	decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
	# pass flattened tensor through decoder
	decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
	# prediction vals
	decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
	# final prediction
	decoder_prediction = tf.argmax(decoder_logits, 2)

	# cross entropy loss
	# one hot encode the target values so we don't rank just differentiate
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),logits=decoder_logits)
	# loss function
	loss = tf.reduce_mean(stepwise_cross_entropy)
	# train it
	train_op = tf.train.AdamOptimizer().minimize(loss)

	batches = helpers_random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_sizes=batch_sizes)
	loss_track = []
	# sess.run(tf.global_variables_initializer())
	sess.run(tf.initialize_all_variables())
	try:
		for batch in range(max_batches):
			fd = next_feed(batches, encoder_inputs, encoder_inputs_length, decoder_targets)
			_, l = sess.run([train_op, loss], feed_dict=fd)
			loss_track.append(l)
			if batch == 0 or batch % batches_in_epoch == 0:
				print('batch {}'.format(batch))
				print('  minibatch loss: {}'.format(sess.run(loss, feed_dict=fd)))
				predict_ = sess.run(decoder_prediction, feed_dict=fd)
				for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
					print('  sample {}:'.format(i + 1))
					print('  input > {}'.format(inp))
					print('  predicted > {}'.format(pred))
					if i >= 2:
						break
				print()
	except KeyboardInterrupt:
		print('training interrupted')

	plt.plot(loss_track)
	plt.show()
	print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))

	return 0


def main():
	run_seq2seq()
	return 0


if __name__ == '__main__':
	main()

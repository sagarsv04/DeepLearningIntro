
import numpy as np # vectorization
import random # generate probability distribution
import tensorflow as tf # machine learning
import datetime # clock training time

# dataset source
# https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset
# Download WikiText-103 raw character level data (~183 MB)

raw_test_data_path = "./wikitext_103/wiki.test.raw"

# NOTE: This code is not memory efficient, Lower the percent_text_data to run on smaller dataset
percent_text_data = 5

# The higher the batch size, the more memory space you'll need.
batch_size = 512
# total iterations
max_steps = 72001
# how often to log?
log_every = 100
# how often to save?
save_every = 6000
hidden_nodes = 1024
# starting text
test_start = 'I am thinking that'
# to save our model
checkpoint_directory = './ckpt'


def read_text_data(raw_data_path):
	# raw_data_path = raw_test_data_path
	with open(raw_data_path, encoding="utf8") as file:
		text_data = file.read()
	print('Text data length in number of characters:', len(text_data))
	if percent_text_data:
		end_point = int((percent_text_data/100)*len(text_data))
		text_data = text_data[:end_point]
		print('Text data length after truncate:', len(text_data))
	# print(text_data[:1000])
	char_list = sorted(list(set(text_data)))
	print('Number of characters:', len(char_list))

	return text_data, char_list


def get_char_to_id(char_list):

	# dictionary that maps each character to a number and vice versa
	char2id = dict((c, i) for i, c in enumerate(char_list))
	id2char = dict((i, c) for i, c in enumerate(char_list))

	return char2id, id2char


def sample(prediction):
	# Given a probability of each character, return a likely character, one-hot encoded
	# our prediction will give us an array of probabilities of each character
	# we'll pick the most likely and one-hot encode it
	# Samples are uniformly distributed over the half-open interval
	r = random.uniform(0,1)
	#store prediction char
	s = 0
	# since length > indices starting at 0
	char_id = len(prediction) - 1
	# for each char prediction probabilty
	for i in range(len(prediction)):
		# assign it to S
		s += prediction[i]
		# check if probability greater than our randomly generated one
		if s >= r:
			# if it is, thats the likely next char
			char_id = i
			break
	# dont try to rank, just differentiate
	# initialize the vector
	char_one_hot = np.zeros(shape=[char_size])
	# that characters ID encoded
	# https://image.slidesharecdn.com/latin-150313140222-conversion-gate01/95/representation-learning-of-vectors-of-words-and-phrases-5-638.jpg?cb=1426255492
	char_one_hot[char_id] = 1.0
	return char_one_hot


def get_x_y_data(text_data, char_size):
	# vectorize our data to feed it into model
	len_per_section = 50
	skip = 2
	sections = []
	next_chars = []
	# fill sections list with chunks of text_data, every 2 characters create a new 50
	# character long section
	# because we are generating it at a character level
	for i in range(0, len(text_data) - len_per_section, skip):
		# i = 0
		sections.append(text_data[i: i + len_per_section])
		next_chars.append(text_data[i + len_per_section])
	# Vectorize input and output
	# matrix of section length by num of characters
	"""
	the issue is that creating everything as one hot encoded for the entire dataset at once consumes a lot of memory.
	So lets just store the indices for the characters and only convert the matrices to one-hot form when we need to process them.
	In future we can create a pickle file and append X and Y one at a time.
	While feeding to model read the pickle file and feed batch wise
	"""
	X = np.zeros((len(sections), len_per_section, char_size))
	# label column for all the character id's, still zero
	Y = np.zeros((len(sections), char_size))
	# for each char in each section, convert each char to an ID
	# for each section convert the labels to ids
	for i, section in enumerate(sections):
		# i, section = 0, sections[0]
		for j, char in enumerate(section):
			# j, char = 5, section[5]
			X[i, j, char2id[char]] = 1
		Y[i, char2id[next_chars[i]]] = 1

	return X, Y


def create_computation_graph(batch_size, len_per_section, char_size):


	return graph


def create_train_our_model( X, Y, batch_size, len_per_section, char_size):
	# build our model time
	graph = tf.Graph()
	# if multiple graphs, but none here jsut one
	with graph.as_default():
		###########
		#Prep
		###########
		#Variables and placeholders
		#global_step refer to the number of batches seen by the graph.
		#Everytime a batch is provided, the weights are updated in the
		#direction that minimizes the loss. global_step just keeps track
		#of the number of batches seen so far starts off as 0
		global_step = tf.Variable(0)

		#data tensor shape feeding in sections
		data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
		#labels
		labels = tf.placeholder(tf.float32, [batch_size, char_size])

		#An LSTM RNN (Long Short Term Memory), consists of 3 gates and an internal state,
		#This enables the LSTM to capture long-term dependencies.
		#http://suriyadeepan.github.io/2017-02-13-unfolding-rnn-2/
		#lets build weights and biases for each of the 3 gates and then for the cell state

		#tf variables
		#Since we need the weights and biases for our model.
		#We could imagine treating these like additional inputs,
		#but TensorFlow has an even better way to handle it: Variable
		#A Variable is a modifiable tensor that lives in TensorFlow's graph of
		#interacting operations. It can be used and even modified by the computation.
		#For machine learning applications, one generally has the model parameters be Variables.

		#Prep LSTM Operation
		#Input gate: weights for input, weights for previous output, and bias

		#tf truncated normal
		#Outputs random values from a truncated normal distribution.
		#The generated values follow a normal distribution with specified mean and
		#standard deviation, except that values whose magnitude is more than 2 standard deviations
		#from the mean are dropped and re-picked.
		#basically randomly initialized values here

		#biases act as an anchor

		w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
		w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
		b_i = tf.Variable(tf.zeros([1, hidden_nodes]))
		#Forget gate: weights for input, weights for previous output, and bias
		w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
		w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
		b_f = tf.Variable(tf.zeros([1, hidden_nodes]))
		#Output gate: weights for input, weights for previous output, and bias
		w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
		w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
		b_o = tf.Variable(tf.zeros([1, hidden_nodes]))
		#Memory cell: weights for input, weights for previous output, and bias
		w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
		w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
		b_c = tf.Variable(tf.zeros([1, hidden_nodes]))

		#LSTM Cell
		# given input, output, external state, it will return output and state
		# output starts off empty, LSTM cell calculates it

		#Since, we have two kinds of states - the internal state ct
		#and the (exposed) external state st, and since we need both of
		#them for the subsequent sequential operations, we combine them
		#into a tensor at each step, and pass them as input to the next
		#step. This tensor is unpacked into st_1 and ct_1 at the beginning of each step.

		def lstm(i, o, state):

			#these are all calculated seperately, no overlap until....
			#(input * input weights) + (output * weights for previous output) + bias
			input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
			#(input * forget weights) + (output * weights for previous output) + bias
			forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
			#(input * output weights) + (output * weights for previous output) + bias
			output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
			#(input * internal state weights) + (output * weights for previous output) + bias
			memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

			#...now! multiply forget gate * given state    +  input gate * hidden state
			state = forget_gate * state + input_gate * memory_cell
			#squash that state with tanh nonlin (Computes hyperbolic tangent of x element-wise)
			#multiply by output
			output = output_gate * tf.tanh(state)
			#return
			return output, state

		###########
		#Operation
		###########
		#LSTM
		#both start off as empty, LSTM will calculate this
		output = tf.zeros([batch_size, hidden_nodes])
		state = tf.zeros([batch_size, hidden_nodes])

		#unrolled LSTM loop
		#for each input set
		for i in range(len_per_section):
			# i = 1
			#calculate state and output from LSTM
			output, state = lstm(data[:, i, :], output, state)
			#to start,
			if i == 0:
				#store initial output and labels
				outputs_all_i = output
				labels_all_i = data[:, i+1, :]
			#for each new set, concat outputs and labels
			elif i != len_per_section - 1:
				#concatenates (combines) vectors along a dimension axis, not multiply
				outputs_all_i = tf.concat([outputs_all_i, output], 0)
				labels_all_i = tf.concat([labels_all_i, data[:, i+1, :]], 0)
			else:
				#final store
				outputs_all_i = tf.concat([outputs_all_i, output], 0)
				labels_all_i = tf.concat([labels_all_i, labels], 0)

		#Classifier
		#The Classifier will only run after saved_output and saved_state were assigned.

		#calculate weight and bias values for the network
		#generated randomly given a size and distribution
		w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
		b = tf.Variable(tf.zeros([char_size]))
		#Logits simply means that the function operates on the unscaled output
		#of earlier layers and that the relative scale to understand the units
		#is linear. It means, in particular, the sum of the inputs may not equal 1,
		#that the values are not probabilities (you might have an input of 5).
		logits = tf.matmul(outputs_all_i, w) + b

		#logits is our prediction outputs, lets compare it with our labels
		#cross entropy since multiclass classification
		#computes the cost for a softmax layer
		#then Computes the mean of elements across dimensions of a tensor.
		#average loss across all values
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_all_i, logits=logits))

		#Optimizer
		#minimize loss with graident descent, learning rate 10,  keep track of batches
		optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)

		###########
		#Test
		###########
		#test_data = tf.placeholder(tf.float32, shape=[1, char_size])
		#test_output = tf.Variable(tf.zeros([1, hidden_nodes]))
		#test_state = tf.Variable(tf.zeros([1, hidden_nodes]))

		#Reset at the beginning of each test
		#reset_test_state = tf.group(test_output.assign(tf.zeros([1, hidden_nodes])),
									#test_state.assign(tf.zeros([1, hidden_nodes])))

		#LSTM
		#test_output, test_state = lstm(test_data, test_output, test_state)
		#test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)
	# timew to train the model, initialize a session with a graph
	with tf.Session(graph=graph) as sess:
		# sess = tf.Session(graph=graph)
		# sess.close()
		# sess.graph
		# standard init step
		# tf.global_variables_initializer().run(session=sess)
		tf.global_variables_initializer().run()
		offset = 0
		saver = tf.train.Saver()
		# for each training step
		for step in range(max_steps):
			# step = 0
			# starts off as 0
			offset = offset % len(X)
			#calculate batch data and labels to feed model iteratively
			if offset <= (len(X) - batch_size):
				#first part
				batch_data = X[offset: offset + batch_size]
				# len(batch_data[0][0])
				batch_labels = Y[offset: offset + batch_size]
				# len(batch_labels[0])
				offset += batch_size
			#until when offset  = batch size, then we
			# labels
			else:
				#last part
				to_add = batch_size - (len(X) - offset)
				batch_data = np.concatenate((X[offset: len(X)], X[0: to_add]))
				batch_labels = np.concatenate((Y[offset: len(X)], Y[0: to_add]))
				offset = to_add

			#optimize!!
			_, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})

			if step % 10 == 0:
				print('Training loss at step %d: %.2f (%s)' % (step, training_loss, datetime.datetime.now()))

				if step % save_every == 0:
					saver.save(sess, checkpoint_directory + '/model', global_step=step)

	return 0


def run_generator(text_data, char2id, id2char, char_size):

	X, Y = get_x_y_data(text_data, char_size)
	# Create a checkpoint directory
	if tf.gfile.Exists(checkpoint_directory):
		tf.gfile.DeleteRecursively(checkpoint_directory)
	tf.gfile.MakeDirs(checkpoint_directory)
	print('Training data size:', len(X))
	print('Approximate steps per epoch:', int(len(X)/batch_size))
	create_train_our_model( X, Y, batch_size, len_per_section, char_size)




	return 0



def main():
	text_data, char_list = read_text_data(raw_test_data_path)
	char2id, id2char = get_char_to_id(char_list)
	run_generator(text_data, char2id, id2char, char_size = len(char_list))
	return 0


if __name__ == '__main__':
	main()

import numpy as np # matrix math
import tensorflow as tf # machine learning
import matplotlib.pyplot as plt # plotting
import time #lets clock training time..
import os

# Import MINST data
# The MNIST data is split into three parts: 55,000 data points of training data
# 10,000 points of test data and 5,000 points of validation data
# very MNIST data point has two parts: an image of a handwritten digit
# and a corresponding label.
# We'll call the images "x" and the labels "y".
# Both the training set and test set contain images and their corresponding labels;
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./temp/data/", one_hot=True)


# Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:
n_pixels = 28*28
# Input to the graph -- Tensorflow's MNIST images are (1, 784) vectors
# x isn’t a specific value.
# It’s a placeholder, a value that we’ll input when we ask TensorFlow
# to run a computation. We want to be able to input any number of MNIST images,
# each flattened into a 784-dimensional vector. We represent this as a 2-D tensor of
# floating-point numbers
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

# layer creation functions
# we could do this inline but cleaner to wrap it in respective functions
# represent the strength of connections between units.

def weight_variable(shape, name):
	# Outputs random values from a truncated normal distribution.
	# truncated means the value is either bounded below or above (or both)
	initial = tf.truncated_normal(shape, stddev=0.1)
	# A Variable is a modifiable tensor that lives in TensorFlow’s graph of
	# interacting operations. It can be used and even modified by the computation.
	# For machine learning applications, one generally has the model parameters
	# be Variables.
	return tf.Variable(initial, name=name)

# Bias nodes are added to increase the flexibility of
# the model to fit the data. Specifically, it allows the
# network to fit the data when all input features are equal to 00,
# and very likely decreases the bias of the fitted values elsewhere in the data space

def bias_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

# Neurons in a fully connected layer have full connections to
# all activations in the previous layer, as seen in regular Neural Networks.
# Their activations can hence be computed with a matrix multiplication followed by a
# bias offset.

def FC_layer(X, W, b):
	return tf.matmul(X, W) + b

# encoder
# our VAE model can parse the information spread thinly over the high-dimensional
# observed world of pixels, and condense the most meaningful features into a
# structured distribution over reduced (20) latent dimensions

# latent = embedded space, we just see latent used in stochastic models in papers a lot
# latent means not directly observed but are rather inferred
latent_dim = 20
# num neurons
h_dim = 500

# layer 1
W_enc = weight_variable([n_pixels, h_dim], 'W_enc')
b_enc = bias_variable([h_dim], 'b_enc')
# tanh activation function to replicate original model
# The tanh function, a.k.a. hyperbolic tangent function,
# is a rescaling of the logistic sigmoid, such that its outputs range from -1 to 1.
# tanh or sigmoid? Whatever avoids the vanishing gradient problem!
h_enc = tf.nn.tanh(FC_layer(X, W_enc, b_enc))

# layer 2
W_mu = weight_variable([h_dim, latent_dim], 'W_mu')
b_mu = bias_variable([latent_dim], 'b_mu')
mu = FC_layer(h_enc, W_mu, b_mu) #mean

# instead of the encoder generating a vector of real values,
# it will generate a vector of means and a vector of standard deviations.
# for reparamterization
W_logstd = weight_variable([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = FC_layer(h_enc, W_logstd, b_logstd)

# reparameterization trick - lets us backpropagate successfully
# since normally gradient descent expects deterministic nodes
# and we have stochastic nodes
# distribution
noise = tf.random_normal([1, latent_dim])
# sample from the standard deviations (tf.exp computes exponential of x element-wise)
# and add the mean
# this is our latent variable we will pass to the decoder
z = mu + tf.multiply(noise, tf.exp(.5*logstd))
# The greater standard deviation on the noise added,
# the less information we can pass using that one variable.
# The more efficiently we can encode the original image,
# the higher we can raise the standard deviation on our gaussian until it reaches one.
# This constraint forces the encoder to be very efficient,
# creating information-rich latent variables.
# This improves generalization, so latent variables that we either randomly generated,
# or we got from encoding non-training images, will produce a nicer result when decoded.

# decoder

# layer 1
W_dec = weight_variable([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')
# pass in z here (and the weights and biases we just defined)
h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))


# layer 2, using the original n pixels here since thats the dimensiaonlty
# we want to restore our data to
W_reconstruct = weight_variable([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')
# 784 bernoulli parameters output
reconstruction = tf.nn.sigmoid(FC_layer(h_dec, W_reconstruct, b_reconstruct))


# lets define our loss function

# variational lower bound

# add epsilon to log to prevent numerical overflow
# Information is lost because it goes from a smaller to a larger dimensionality.
# How much information is lost? We measure this using the reconstruction log-likelihood
# This measure tells us how effectively the decoder has learned to reconstruct
# an input image x given its latent representation z.
log_likelihood = tf.reduce_sum(X*tf.log(reconstruction + 1e-9)+(1 - X)*tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
# KL Divergence
# If the encoder outputs representations z that are different
# than those from a standard normal distribution, it will receive
# a penalty in the loss. This regularizer term means
# ‘keep the representations z of each digit sufficiently diverse’.
# If we didn’t include the regularizer, the encoder could learn to cheat
# and give each datapoint a representation in a different region of Euclidean space.
KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)

# This allows us to use stochastic gradient descent with respect to the variational parameters
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

# init all variables and start the session!
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# store value for these 3 terms so we can plot them later
variational_lower_bound_array = []
log_likelihood_array = []
KL_term_array = []


def training():
	num_iterations = 1000000
	recording_interval = 1000

	iteration_array = [i*recording_interval for i in range(int(num_iterations/recording_interval))]
	for i in range(num_iterations):
		# np.round to make MNIST binary
		# get first batch (200 digits)
		x_batch = np.round(mnist.train.next_batch(200)[0])
		# run our optimizer on our data
		sess.run(optimizer, feed_dict={X: x_batch})
		if (i%recording_interval == 0):
			# every 1K iterations record these values
			vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
			if (i%10000) == 0:
				print("Iteration: {0}, Loss: {1}".format(i, vlb_eval))
			variational_lower_bound_array.append(vlb_eval)
			log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
			KL_term_array.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))

	return iteration_array


def plot_loss():
	plt.figure()
	# for the number of iterations we had
	# plot these 3 terms
	plt.plot(iteration_array, variational_lower_bound_array)
	plt.plot(iteration_array, KL_term_array)
	plt.plot(iteration_array, log_likelihood_array)
	plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
	plt.title('Loss per iteration')

	return 0


def plot_numbers():

	num_pairs = 10
	image_indices = np.random.randint(0, 200, num_pairs)
	#Lets plot 10 digits
	for pair in range(num_pairs):
		#reshaping to show original test image
		x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
		plt.figure()
		x_image = np.reshape(x, (28,28))
		plt.subplot(121)
		plt.imshow(x_image)
		#reconstructed image, feed the test image to the decoder
		x_reconstruction = reconstruction.eval(feed_dict={X: x})
		#reshape it to 28x28 pixels
		x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
		#plot it!
		plt.subplot(122)
		plt.imshow(x_reconstruction_image)

	return 0


def main():

	training()
	plot_loss()
	plot_numbers()

	return 0


if __name__ == '__main__':
	main()

from datetime import datetime
import numpy as np
import pandas as pd
import copy
import os
import re


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\HowToDoMath')


def load_from_file(filename):
	# filename = "earthquake_dataset.csv"

	df_dataset = pd.read_csv(filename)
	# df_dataset.head()
	date = df_dataset['Date'].tolist()
	latitude = df_dataset['Latitude'].tolist()
	longitude = df_dataset['Longitude'].tolist()
	magnitude = df_dataset['Magnitude'].tolist()

	# elements = date[0]
	date = [datetime.strptime(element,"%m/%d/%Y") for element in date]

	return np.array(date), np.float32(latitude), np.float32(longitude), np.float32(magnitude)


def normalize_date(array):

	min_data = min(array)
	max_data = max(array)
	delta = max_data - min_data

	return np.float32([(d - min_data).total_seconds() / delta.total_seconds() for d in array])


def normalize_cord(latitude, longitude):

	rad_lat = np.deg2rad(latitude)
	rad_lon = np.deg2rad(longitude)

	x = np.cos(rad_lat) * np.cos(rad_lon)
	y = np.cos(rad_lat) * np.sin(rad_lon)
	z = np.sin(rad_lat)

	return x, y, z


def vectorize(date, latitude, longitude):

	return np.concatenate(normalize_cord(latitude, longitude) +
			(normalize_date(date),))\
			.reshape((4, len(date)))\
			.swapaxes(0, 1)


def sigmoid(x, deriv=False):

	if deriv:
		return x * (1 - x)

	return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):

	if deriv:
		return np.ones_like(x) * (x > 0)

	return x * (x > 0)


def new_parameters(x, x_min, x_max, radius):

	alpha = 2 * np.random.random() - 1
	new_x = x + radius * alpha

	if new_x < x_min:
		return x_min
	elif new_x > x_max:
		return x_max

	return new_x


def get_batch(batch_size, X, Y):

	if X.shape[0] % batch_size != 0:
		print("Warning !! the full set will not be executed because of a poor choice of batch_size")

	for i in range(X.shape[0] // batch_size):
		yield X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size]


def gen_random_batch(batch_size, X, Y):

	while True:
		index = np.arange(X.shape[0])
		np.random.shuffle(index)

		s_X, s_Y = X[index], Y[index]
		for i in range(X.shape[0] // batch_size):
			yield (X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size])


def main():

	# load and prepare data
	date, latitude, longitude, magnitude = load_from_file("./earthquake_dataset.csv")
	data_size = len(date)
	vectorsX, vectorsY = vectorize(date, latitude, longitude), magnitude.reshape((data_size, 1))

	# split vectors into train and test sets
	test_set_size = int(0.1 * data_size)
	index = np.arange(data_size)
	np.random.shuffle(index)
	trainX, trainY = vectorsX[index[test_set_size:]], vectorsY[index[test_set_size:]]
	testX, testY = vectorsX[index[:test_set_size]], vectorsY[index[:test_set_size]]
	# len(trainX)
	# len(testX)
	# randomly initialize our weights with mean 0
	syn0_origin = 2 * np.random.random((trainX.shape[1], 32)) - 1
	syn1_origin = 2 * np.random.random((32, trainY.shape[1])) - 1

	# hyperparameters
	best_error = 9999
	best_learning_rate_log = -3
	best_momentum = 0.9
	best_batch_size = 64
	best_max_epochs_log = 4
	learning_rate_log = None
	momentum = None
	batch_size = None
	max_epochs_log = None

	for i in range(50):
		# i = 0
		# Hyperparameters
		learning_rate_log = new_parameters(best_learning_rate_log, -5, -1, 0.5)  # log range from 0.0001 to 0.1
		momentum = new_parameters(best_momentum, 0.5, 0.95, 0.1)  # linear range from 0.5 to 0.9
		batch_size = np.int64(new_parameters(best_batch_size, 10, 128, 10))  # linear range from 10 to 128
		max_epochs_log = new_parameters(best_max_epochs_log, 3, 5, 0.5)  # log range from 1000 to 100000

		learning_rate = np.power(10, learning_rate_log)
		max_epochs = np.int64(np.power(10, max_epochs_log))

		# display hyperparameters
		print("iteration:{0} learning rate:{1} momentum:{2} batch size:{3} max epochs:{4}".format(i, learning_rate, momentum, batch_size, max_epochs))

		# reset weight
		syn0 = copy.deepcopy(syn0_origin)
		syn1 = copy.deepcopy(syn1_origin)

		# initialize momentum
		momentum_syn0 = np.zeros_like(syn0)
		momentum_syn1 = np.zeros_like(syn1)

		# get batch generator
		batch_gen = gen_random_batch(batch_size, trainX, trainY)

		# Train model
		for j in range(max_epochs):
			# Get Batch
			batch = next(batch_gen)

			# feed forward
			l0 = batch[0]
			l1 = sigmoid(np.dot(l0, syn0))
			l2 = relu(np.dot(l1, syn1))

			# l2 error & delta
			l2_error = batch[1] - l2
			l2_delta = l2_error * relu(l2, deriv=True)

			# l1 error & delta
			l1_error = l2_delta.dot(syn1.T)
			l1_delta = l1_error * sigmoid(l1, deriv=True)

			# momentum
			momentum_syn1 = momentum * momentum_syn1 + l1.T.dot(l2_delta) * learning_rate
			momentum_syn0 = momentum * momentum_syn0 + l0.T.dot(l1_delta) * learning_rate

			# Apply momentum correction
			syn1 += momentum_syn1
			syn0 += momentum_syn0

		# Evaluate model
		current_error = 0
		for batch in get_batch(10, testX, testY):
			# feed forward
			l0 = batch[0]
			l1 = sigmoid(np.dot(l0, syn0))
			l2 = relu(np.dot(l1, syn1))

			# accumulate error
			current_error += np.sum(np.abs(batch[1] - l2))
		current_error /= test_set_size

		print("Error: {0}".format(current_error))

		if current_error < best_error:
			best_error = current_error
			best_learning_rate_log = learning_rate_log
			best_momentum = momentum
			best_batch_size = batch_size
			best_max_epochs_log = max_epochs_log

	print("Best Error: {0}".format(best_error))
	print("Best Learning Rate Log: {0}".format(best_learning_rate_log))
	print("Best Momentum: {0}".format(best_momentum))
	print("Best Batch Size: {0}".format(best_batch_size))
	print("Best Max Epochs Log: {0}".format(best_max_epochs_log))

	return 0


if __name__ == "__main__":
	main()

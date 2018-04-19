
import time
import os
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import datetime
import math, time
import itertools
import quandl
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\PredictStockPrices')


data_path = './S&P500_close_price.csv'
stock_data_path = './google_ohlc.csv'


def plot_results_multiple(predicted_data, true_data, prediction_len):
	# predicted_data, true_data, prediction_len = predictions, y_test, 50
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')

	# Pad the list of predictions to shift it in the graph to it's correct start
	for i, data in enumerate(predicted_data):
		# i, data = 1, predicted_data[1]
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label='Prediction')
		plt.legend()
	plt.show()


def normalise_windows(window_data):
	# window_data = result
	normalised_data = []
	for window in window_data:
		# window = window_data[0]
		normalised_window = [((float(p) / float(window[0])) - 1) for p in window] # p = window[1]
		normalised_data.append(normalised_window)
	return normalised_data


def get_stock_data(stock_data_path, normalized=False):

	df_data =  pd.read_csv(stock_data_path)
	df_data['High'] = df_data['High'] / 1000
	df_data['Open'] = df_data['Open'] / 1000
	df_data['Close'] = df_data['Close'] / 1000
	df_data['Low'] = df_data['Low'] / 1000
	df_data = df_data[['Open', 'High', 'Low', 'Close']]
	if normalized:
		# normalize the data
		pass

	return df_data


def load_data(filename, seq_len, normalise_window, ohcl=False):
	# filename, seq_len, normalise_window, ohcl = data_path, 50, True, False
	# filename, seq_len, normalise_window, ohcl = df_data[::-1], window, False, True
	# seq_len is the rolling window length
	if not ohcl:
		f = open(filename, 'r').read()
		data = f.split('\n')
		amount_of_features = 1
	else:
		amount_of_features = len(filename.columns)
		data = filename.as_matrix()

	sequence_length = seq_len + 1
	result = [] # len(result)
	# loading data
	for index in range(len(data) - sequence_length):
		# index = 0 # data[index]
		result.append(data[index: index + sequence_length])

	if normalise_window and not ohcl:
		result = normalise_windows(result)

	result = np.array(result)

	row = round(0.9 * result.shape[0])
	train = result[:int(row), :] # len(train)
	np.random.shuffle(train)
	x_train = train[:, :-1] # len(x_train)
	y_train = train[:, -1] # len(y_train)
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

	return x_train, y_train, x_test, y_test


def predict_point_by_point(model, data):
	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	predicted = model.predict(data)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted


def predict_sequence_full(model, data, window_size):
	#Shift the window by 1 new prediction each time, re-run predictions on new window
	curr_frame = data[0]
	predicted = []
	for i in xrange(len(data)):
		predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
		curr_frame = curr_frame[1:]
		curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
	return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
	# Predict sequence of 50 steps before shifting prediction run forward by 50 steps
	# model, data, window_size, prediction_len = model, X_test, 50, 50
	prediction_seqs = []
	for i in range(int(len(data)/prediction_len)):   # int(1.7) = 1
		# i = 0
		curr_frame = data[i*prediction_len]
		predicted = []
		for j in range(prediction_len):
			predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
		prediction_seqs.append(predicted)
	return prediction_seqs


def create_lstm_model_one():

	model = Sequential()
	model.add(LSTM(input_shape=(None, 1), units=50, return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(100, return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(units=1))
	model.add(Activation('linear'))

	model.compile(loss='mse', optimizer='rmsprop')

	return model


def create_lstm_model_two(layers):
	# layers = [3,window,1]

	model = Sequential()
	model.add(LSTM(input_shape=(layers[1], layers[0]), units=128, return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(input_shape=(layers[1], layers[0]), units=64, return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(units=layers[0], kernel_initializer='uniform', activation='relu'))

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	return model


def run_on_close_price_csv():

	X_train, y_train, X_test, y_test = load_data(data_path, 50, True, False)
	model = create_lstm_model_one()
	model.fit(x=X_train, y=y_train, batch_size=512, epochs=10, validation_split=0.05, verbose=1)

	predictions = predict_sequences_multiple(model, X_test, 50, 50)
	plot_results_multiple(predictions, y_test, 50)

	return 0


def run_on_stock_csv():

	df_data = get_stock_data(stock_data_path, False)
	window = 5
	# reverse the dataframe order
	# filename, seq_len, normalise_window, ohcl=False
	X_train, y_train, X_test, y_test = load_data(df_data[::-1], window, False, True)
	print("X_train", X_train.shape)
	print("y_train", y_train.shape)
	print("X_test", X_test.shape)
	print("y_test", y_test.shape)
	# model = build_model([3,lag,1])
	model = create_lstm_model_two([X_train.shape[-1], window, 1])

	model.fit(x=X_train, y=y_train, batch_size=512, epochs=10, validation_split=0.1, verbose=1)
	trainScore = model.evaluate(X_train, y_train, verbose=1)
	print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

	testScore = model.evaluate(X_test, y_test, verbose=1)
	print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

	diff=[]
	ratio=[]
	p = model.predict(X_test)
	for u in range(len(y_test)):
		pr = p[u][0]
		ratio.append((y_test[u]/pr)-1)
		diff.append(abs(y_test[u]- pr))

	plt.plot(p,color='red', label='prediction')
	plt.plot(y_test,color='blue', label='y_test')
	plt.legend(loc='upper left')
	plt.show()

	return 0


def main():

	run_on_close_price_csv()
	# run_on_stock_csv()

	return 0


if __name__ == '__main__':
	main()

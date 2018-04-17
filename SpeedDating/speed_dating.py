
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\SpeedDating')

# Thake a look on Overview
# https://www.kaggle.com/annavictoria/speed-dating-experiment
# to download the data use
# https://www.kaggle.com/annavictoria/speed-dating-experiment/data


data_path = './speed_dating_data.csv'

people_dim = 10
hidden = 32
batch_size = 60
epochs = 10000


class Normalizer:
	def __init__(self, training_array):
		self.__mini = np.min(training_array, axis=0)
		maxi = np.max(training_array, axis=0)
		self.__delta = maxi - self.__mini

	def __call__(self, array):
		return np.apply_along_axis(lambda x : (x-self.__mini)/self.__delta, 1, array)


class PCA:
	def get_projection_matrix(self, array, n):
		corr = np.cov(array, rowvar=False)
		v, w = np.linalg.eig(corr)

		index = np.argsort(v)[::-1]
		return w[:, index[:n]]

	def __init__(self, training_array, n):
		self.__proj = self.get_projection_matrix(training_array, n)

	def __call__(self, array):
		return self.__proj.T.dot(array.T).T


class Preprocess:
	def __init__(self, df, dim):
		self.__df = df
		self.__norm = Normalizer(self.__df.values)
		self.__red = PCA(self.__norm(self.__df.values), dim)

	def __call__(self, iid):
		return self.__red(self.__norm(self.__df.loc[iid, :].values))


def extract_database(data_path):

	df = pd.read_csv(data_path)
	# df.head() # df.info()
	# Table of people
	df_p = df.drop_duplicates('iid').set_index('iid').loc[:, 'age':'amb3_1']
	# df_p.head() # df_p.info()
	df_p.loc[:, 'attr1_1':'amb3_1'] = df.loc[:, 'attr1_1':'amb3_1'].replace(to_replace=pd.NaT, value=0.0)

	df_p = df_p.drop(["mn_sat", "undergra", "tuition", "from", "zipcode", "income", "field","field_cd",
					   "career", "attr4_1", "sinc4_1", "intel4_1", "fun4_1", "amb4_1","shar4_1"], axis=1)

	df_p = df_p.dropna(axis=0)

	# Table of meeting and result
	df_m = df.loc[:, ['iid', 'pid', 'match']]
	df_m = df_m[df_m["iid"].isin(list(df_p.index.values))]
	df_m = df_m[df_m["pid"].isin(list(df_p.index.values))]
	df_m = df_m.reset_index(drop=True)

	return df_p, df_m


def gen_profile(df_m, vectorize):
	# Batches generator, provide random batches of given size
	gen = df_m.sample(frac=1).reset_index().iterrows()
	for j in range(epochs):
		iid, pid, match = [], [], []
		for i in range(batch_size):
			try:
				row = next(gen)[1]
			except StopIteration:
				gen = df_m.sample(frac=1).reset_index().iterrows()
				row = next(gen)[1]
			iid.append(int(row.loc['iid']))
			pid.append(int(row.loc['pid']))
			match.append(row.loc['match'])

		yield np.float32(match).reshape([batch_size, 1]), vectorize(iid), vectorize(pid)


def find(df_p,i):
	# Model Testing
	n = 0
	for j in list(df_p.index):
		if i==j:
			return n
		n += 1


def run_speed_dating():

	df_p, df_m = extract_database(data_path)
	red = Preprocess(df_p.loc[:, 'attr1_1':'amb3_1'], people_dim)
	blue = Preprocess(df_p.loc[:, 'date':'yoga'], people_dim)
	vectorize = lambda x : np.concatenate((red(x), blue(x)), axis=1)

	# model definition
	# Placeholders
	iid_ph = tf.placeholder(np.float32, shape=[None, people_dim*2])
	pid_ph = tf.placeholder(np.float32, shape=[None, people_dim*2])
	match_ph = tf.placeholder(np.float32, shape=[None, 1])

	# Model
	cont_ab = tf.concat((iid_ph, pid_ph), axis=1)
	cont_ba = tf.concat((pid_ph, iid_ph), axis=1)

	w_1 = tf.Variable(tf.truncated_normal([people_dim*4, hidden], stddev=0.1))
	b_1 = tf.Variable(tf.truncated_normal([hidden], stddev=0.1))
	l_h_ab = tf.nn.softplus(tf.matmul(cont_ab, w_1) + b_1)
	l_h_ba = tf.nn.softplus(tf.matmul(cont_ba, w_1) + b_1)

	w_2 = tf.Variable(tf.truncated_normal([hidden, 1], stddev=0.1))
	b_2 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
	l_2 = tf.matmul(l_h_ab, w_2) + tf.matmul(l_h_ba, w_2) + b_2
	logits = tf.sigmoid(l_2)

	# Trainer
	print("Total vars: ~{:.2f}k\n".format((4*people_dim*hidden + 2*hidden + 1)/1000))
	global_step = tf.Variable(0, trainable=False)
	learing_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.8, staircase=True)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l_2, labels=match_ph))
	trainer_step = tf.train.MomentumOptimizer(learing_rate, 0.9).minimize(loss, global_step=global_step)

	# Processing

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		# Train the model
		print("Training...")
		i = 0
		for profile in gen_profile(df_m, vectorize):
			trainer_step.run(feed_dict={iid_ph: profile[1], pid_ph: profile[2], match_ph: profile[0]})
			if i % 1000 == 0:
				print(loss.eval(feed_dict={iid_ph: profile[1], pid_ph: profile[2], match_ph: profile[0]}))
			i += 1
		print("\n")

		# Let's find love <3
		# It fill a lookup table, it's not the goal, but it's useful to dispay results
		print("Looking for hope...")
		l_df_p = len(df_p)
		mp = np.ndarray((l_df_p, l_df_p), dtype=np.float32)  # Map of result
		for i in range(l_df_p):
			pid = df_p.index[i]
			for j in range(l_df_p):
				iid = df_p.index[j]
				mp[i, j] = logits.eval(feed_dict={iid_ph: vectorize([iid]), pid_ph: vectorize([pid])})[0, 0]
		print("Done")

	error = 0.0
	for index, rows in df_m.iterrows():
		error += abs(rows["match"] - mp[find(df_p, rows["iid"]), find(df_p, rows["pid"])])
	print("Avg error: {:.2f}%".format(100 * error / len(df_m)))

	# Display the Love map
	plt.imshow(mp)
	plt.show()

	return 0


def main():

	run_speed_dating()

	return 0


if __name__ == '__main__':
	main()

import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
import time
import os
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation
import pylab as pl

import math as mt
import csv
from sparsesvd import sparsesvd #used for matrix factorization
from pylab import *
from scipy.sparse import csc_matrix #used for sparse matrix
from scipy.sparse.linalg import * #used for matrix multiplication


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\Recommender')

# this is 10000 rows clean data with no duplicates
triplets_file = './data/triplets_file.csv'
songs_file = './data/song_data.csv'
# define what percentage of users to use for precision recall calculation
user_sample = 0.05
# constants defining the dimensions of our User Rating Matrix (URM)
MAX_PID = 4
MAX_UID = 5


'''
# If you want the raw unprocessed data
# triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
# songs_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
'''


def load_data():

	# read userid-songid-listen_count triplets
	song_df_1 = pandas.read_csv(triplets_file)
	# song_df_1.head() # len(song_df_1)
	song_df_2 =  pandas.read_csv(songs_file)
	# song_df_2.head() # len(song_df_2)
	# merge the two dataframes above to create input dataframe for recommender systems
	song_df = pandas.merge(song_df_1, song_df_2, on="song_id", how="left")
	# song_df.head() # len(song_df)

	return song_df


def manipulate_data(song_df):
	# merge song title and artist_name columns to make a merged column
	song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

	song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
	grouped_sum = song_grouped['listen_count'].sum()
	song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100

	song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

	users = song_df['user_id'].unique()
	# len(users)
	print("Number of unique users: {0}".format(len(users)))
	songs = song_df['song'].unique()
	# len(songs)
	print("Number of unique songs: {0}".format(len(songs)))

	return song_df, users, songs


def get_train_test_data(song_df):

	train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
	# len(train_data) , len(test_data)
	return train_data, test_data


def popularity_based(train_data, test_data, users, user_nos):
	'''
	user_nos is the ith user in the users list

	'''
	# create an instance of popularity based recommender class
	pm_model = Recommenders.popularity_recommender_py()
	pm_model.create(train_data, 'user_id', 'song')

	# use the popularity model to make some predictions
	user_id = users[user_nos]
	pm_model.recommend(user_id)
	print("Below are the list of songs recommended for user_id: {0}\n".format(user_id))
	for song in pm_model.recommend(user_id)['song']:
		# song = pm_model.recommend(user_id)['song'][0]
		print(song)

	return pm_model


def similarity_based(train_data, test_data, users, user_nos):
	'''
	user_nos is the ith user in the users list

	'''
	# create an instance of similarity based recommender class
	is_model = Recommenders.item_similarity_recommender_py()
	is_model.create(train_data, 'user_id', 'song')

	user_id = users[user_nos]
	user_items = is_model.get_user_items(user_id)

	print("------------------------------------------------------------------------------------")
	print("Training data songs for the user userid: %s:" % user_id)
	print("------------------------------------------------------------------------------------")

	for user_item in user_items:
		print(user_item)

	print("----------------------------------------------------------------------")
	print("Recommendation process going on:")
	print("----------------------------------------------------------------------")

	# recommend songs for the user using personalized model
	user_recom = is_model.recommend(user_id)
	for song in user_recom['song']:
		# song = pm.recommend(user_id)['song'][0]
		print(song)

	song = user_recom.iloc[0]['song']
	print("----------------------------------------------------------------------")
	print("Similar recommendation to song: %s" % song)
	print("----------------------------------------------------------------------")

	similar_items = is_model.get_similar_items([song])
	for song in similar_items['song']:
		# song = pm.recommend(user_id)['song'][0]
		print(song)

	return is_model


def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m2_recall_list, m2_label):
	pl.clf()
	pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
	pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
	pl.xlabel('Recall')
	pl.ylabel('Precision')
	pl.ylim([0.0, 0.20])
	pl.xlim([0.0, 0.20])
	pl.title('Precision-Recall curve')
	#pl.legend(loc="upper right")
	pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
	pl.show()

	return 0


def cal_precision_recall(train_data, test_data, pm_model, is_model):

	start = time.time()
	# instantiate the precision_recall_calculator class
	pr = Evaluation.precision_recall_calculator(test_data, train_data, pm_model, is_model)
	# call method to calculate precision and recall values
	(pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)
	end = time.time()
	print(end - start)

	print("Plotting precision recall curves.")

	plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
					  ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")

	return 0


def computeSVD(urm, K):
	# compute SVD of the user ratings matrix
    U, s, Vt = sparsesvd(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):
	# compute estimated rating for the test user
    rightTerm = S*Vt

    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        #we convert the vector to dense format in order to get the indices
        #of the movies with the best estimated ratings
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
    return recom


def matrix_factorization_based():
	# used in SVD calculation (number of latent factors)
	K=2
	# initialize a sample user rating matrix
	urm = np.array([[3, 1, 2, 3],[4, 3, 4, 3],[3, 2, 1, 5], [1, 6, 5, 2], [5, 0,0 , 0]])
	urm = csc_matrix(urm, dtype=np.float32)

	# compute SVD of the input user ratings matrix
	U, S, Vt = computeSVD(urm, K)

	# test user set as user_id 4 with ratings [0, 0, 5, 0]
	uTest = [4]
	print("User id for whom recommendations are needed: %d" % uTest[0])

	# get estimated rating for test user
	print("Predictied ratings:")
	uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)
	print(uTest_recommended_items)

	# plot all the users
	print("Matrix Dimensions for U")
	print(U.shape)

	for i in range(0, U.shape[0]):
	    plot(U[i,0], U[i,1], marker = "*", label="user"+str(i))

	for j in range(0, Vt.T.shape[0]):
	    plot(Vt.T[j,0], Vt.T[j,1], marker = 'd', label="item"+str(j))

	legend(loc="upper right")
	title('User vectors in the Latent semantic space')
	ylim([-0.7, 0.7])
	xlim([-0.7, 0])
	show()

	return 0


def main():
	song_df = load_data()
	song_df, users, songs = manipulate_data(song_df)
	train_data, test_data = get_train_test_data(song_df)

	pm_model = popularity_based(train_data, test_data, users, 5)

	is_model = similarity_based(train_data, test_data, users, 5)

	cal_precision_recall(train_data, test_data, pm_model, is_model)
	matrix_factorization_based()

	return 0


if __name__ == '__main__':
	main()

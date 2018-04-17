
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\DimnensionalityReduction')


# Dimensionaltiy reduction reasons
# 1. Space efficiency - makes data small
# 2. Computing efficiency It fastens the time required for performing same computations.
# 3. Visualizations!
# Less dimensions leads to less computing, also less dimensions can allow usage of algorithms unfit for a large number of dimensions like linear regression

'''
Compare PCA, T-SNE, LDA.

# pros
T-SNE gives better visualizations

# on every other mark, efifiecny accuracy, PCA is better

# PCA is label agnostic -- it treats the entire data set as a whole.
# LDA, on the other hand, tries to explicitly model difference between classes(labels) within the data.

# PCA performs unsupervised transformation, while LDA is supervised.

Best visualizations
T-SNE

Best generic dim r method
PCA

Best for supervised
LDA

T-SNE
Step 1 - Compute similarity matrix between all feature vectors
Step 2 - Compute similarity matrix from map points
Step 3 - Use gradient descent to minimimze distance between matrices

LDA (Linear Discriminant Analysis)

Similar to PCA except Compute the M mean vectors for the different classes from the dataset instead of whole dataset.
In PCA we take the whole dataset consisting of dd-dimensional samples ignoring the class labels
Compute the dd-dimensional mean vector (i.e., the means for every dimension of the whole dataset).
'''

# to know more on PCA head on to this link
# https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/

# to know more on t-SNE head on to this link
# https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm


def run_pca():

	# random seed for consistency, debugging same results every time
	np.random.seed(1)
	# We'll first create 2 classes
	# each with 3 features, create classes random sampled 3 X 20 data set

	# Draw random samples from a multivariate normal distribution.
	# covariance measures the degree to which two variables are linearly associated.
	mu_vec1 = np.array([0,0,0])  # sample mean
	cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # sample covariance
	class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T

	mu_vec2 = np.array([1,1,1]) # sample mean
	cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #sample covariance
	class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T

	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10
	ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
	ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, color='green', alpha=0.5, label='class2')
	ax.legend(loc='upper right')
	plt.show()

	# make it one big dataset
	# 3 x 40 still 3 features
	all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
	# all_samples.T

	# compute the d dimensional mean vector, to help compute covariance matrix
	# mean for each feature
	mean_x = np.mean(all_samples[0,:])
	mean_y = np.mean(all_samples[1,:])
	mean_z = np.mean(all_samples[2,:])

	# 3D mean vector
	mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
	print('Mean Vector:\n', mean_vector)

	# compute the covariance matrix
	# measures relationship between each feature
	cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
	print('Covariance Matrix:\n', cov_mat)

	# eigenvectors and eigenvalues for the from the scatter matrix
	eig_val_sc, eig_vec_sc = np.linalg.eig(cov_mat)

	for i in range(len(eig_val_sc)):
	    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
	    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
	    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))

	# sort eigenvector by decreasing value
	# make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort()
	eig_pairs.reverse()

	# Visually confirm that the list is correctly sorted by decreasing eigenvalues
	print('Eigenvalue in order')
	for i in eig_pairs: print(i[0])

	# choose k eigenvectos w largest eigenvalues to form d x k matrix

	matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
	print('Matrix W:\n', matrix_w)

	# use d x k to transform samples to new subspace
	# dot product between orignal matrix and eigen pairs
	transformed = matrix_w.T.dot(all_samples)
	assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."

	plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=8, color='green', alpha=0.5, label='class1')
	plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=8, color='red', alpha=0.5, label='class2')
	plt.xlim([-5,5])
	plt.ylim([-5,5])
	plt.xlabel('x_values')
	plt.ylabel('y_values')
	plt.legend()
	plt.title('Transformed samples with class labels')
	plt.show()

	return 0


def main():
	run_pca()

	return 0



if __name__ == '__main__':
	main()

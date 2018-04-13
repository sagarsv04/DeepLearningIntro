import numpy as np
from matplotlib import pyplot as plt


def nonlin(x,deriv=False):
	# if weights are given then calculate derivative of data
	if(deriv==True):
	    return x*(1-x)
	# if weights not given then just return sigmoid funcion computation
	return 1/(1+np.exp(-x))


def math_demo():

	X = np.array([[0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1]])

	y = np.array([[0],
				[1],
				[1],
				[0]])

	np.random.seed(1)

	# randomly initialize our weights with mean 0
	syn0 = 2*np.random.random((3,4)) - 1
	syn1 = 2*np.random.random((4,1)) - 1


	for j in range(60000*2):
		# j = 0
		# Feed forward through layers 0, 1, and 2
		k0 = X
		k1 = nonlin(np.dot(k0,syn0))
		k2 = nonlin(np.dot(k1,syn1))

		# how much did we miss the target value?
		k2_error = y - k2

		if (j% 1000) == 0:
			print("Error:" + str(np.mean(np.abs(k2_error))))

		# in what direction is the target value?
		# were we really sure? if so, don't change too much.
		k2_delta = k2_error*nonlin(k2,deriv=True)

		# how much did each k1 value contribute to the k2 error (according to the weights)?
		k1_error = k2_delta.dot(syn1.T)

		# in what direction is the target k1?
		# were we really sure? if so, don't change too much.
		k1_delta = k1_error * nonlin(k1,deriv=True)

		syn1 += k1.T.dot(k2_delta)
		syn0 += k0.T.dot(k1_delta)

	return 0


def main():

	math_demo()

	return 0


if __name__ == '__main__':
	main()

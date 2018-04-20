from __future__ import print_function

import time
import numpy as np
from PIL import Image
import cv2
import os

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\GenerateArt')


height = 512
width = 512

content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0
iterations = 10

content_image_path = './images/elephant.jpg'
style_image_path = './images/styles/wave.jpg'
content_video_path = './images/sample.mp4'
style_image_paths = [style_image_path]


def load_images():
	# Load the images and resize them
	style_image = Image.open(style_image_path)
	content_image = Image.open(content_image_path)
	style_image = style_image.resize((height, width))
	content_image = content_image.resize((height, width))

	return content_image, style_image


def load_video():
	# Load the video in frames and resize them
	frames = []
	vc = cv2.VideoCapture(content_video_path)

	if vc.isOpened():
		rval , frame = vc.read()
	else:
		rval = False
	while rval:
		frames.append(frame)
		rval, frame = vc.read()
	vc.release()
	print('Frames collected:', len(frames))

	content_images = []
	for frame in frames:
		content_image = Image.fromarray(frame)
		content_image = content_image.resize((height, width))
		content_images.append(content_image)

	style_images = []
	for style_img_path in style_image_paths:
		style_image = Image.open(style_img_path)
		style_image = style_image.resize((height, width))
		style_images.append(style_image)

	return content_images, style_images


def get_image_array(image):

	image_array = np.asarray(image, dtype='float32')
	image_array = np.expand_dims(image_array, axis=0)
	# Subtract the mean RGB
	image_array[:, :, :, 0] -= 103.939
	image_array[:, :, :, 1] -= 116.779
	image_array[:, :, :, 2] -= 123.68
	# Flip the ordering of the multi-dimensional array from RGB to BGR
	image_array = image_array[:, :, :, ::-1]

	return image_array


def get_video_images(frames):

	video_images = []
	for frame in frames:
		frame_array = np.asarray(frame, dtype='float32')
		frame_array = np.expand_dims(frame_array, axis=0)
		# Subtract the mean RGB
		frame_array[:, :, :, 0] -= 103.939
		frame_array[:, :, :, 1] -= 116.779
		frame_array[:, :, :, 2] -= 123.68
		# Flip the ordering of the multi-dimensional array from RGB to BGR
		frame_array = frame_array[:, :, :, ::-1]
		frame_image = backend.variable(frame_array)
		video_images.append(frame_image)

	return video_images


def get_image(image_array, create_image=False):
	# we need to subject our output image to the inverse of the transformation we did to our input images before it makes sense.
	image_array = image_array.reshape((height, width, 3))
	image_array = image_array[:, :, ::-1]
	image_array[:, :, 0] += 103.939
	image_array[:, :, 1] += 116.779
	image_array[:, :, 2] += 123.68
	image_array = np.clip(image_array, 0, 255).astype('uint8')
	if create_image:
		image = Image.fromarray(image_array)
	else:
		image =	image_array

	return image


def content_loss(content, combination):
	# The content loss is the (scaled, squared) Euclidean distance between feature representations of the content and combination images.
	return backend.sum(backend.square(combination - content))


def gram_matrix(x):
	features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
	gram = backend.dot(features, backend.transpose(features))
	return gram


def style_loss(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = height * width
	return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
	a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
	b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
	return backend.sum(backend.pow(a + b, 1.25))


def eval_loss_and_grads(x, f_outputs):
	x = x.reshape((1, height, width, 3))
	outs = f_outputs([x])
	loss_value = outs[0]
	grad_values = outs[1].flatten().astype('float64')
	return loss_value, grad_values


class Evaluator(object):
	# We then introduce an Evaluator class that computes loss and gradients in one pass while retrieving them via two separate functions
	# total_variation_loss and eval_loss_and_grads.
	# This is done because scipy.optimize requires separate functions for loss and gradients, but computing them separately would be inefficient.
	def __init__(self, f_outputs):
		self.loss_value = None
		self.grads_values = None
		self.f_outputs = f_outputs

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x, self.f_outputs)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


def run_style_transfer():

	content_image, style_image = load_images()
	content_array = get_image_array(content_image)
	style_array  = get_image_array(style_image)

	# define variables in Keras' backend (the TensorFlow graph).
	content_image = backend.variable(content_array)
	style_image = backend.variable(style_array)
	# combination image that retains the content of the content image while incorporating the style of the style image
	combination_image = backend.placeholder((1, height, width, 3))

	# Finally, we concatenate all this image data into a single tensor that's suitable for processing by Keras' VGG16 model.
	input_tensor = backend.concatenate([content_image, style_image, combination_image], axis=0)

	# Reuse a model pre-trained for image classification to define loss functions
	# weights='imagenet' will use pretrained weights
	# NOTE: it will download vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 of size ~ 58MB if not found in directory
	model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
	# Let's make a list of these names so that we can easily refer to individual layers later.
	layers = dict([(layer.name, layer.output) for layer in model.layers])
	# We begin by initialising the total loss to 0 and adding to it in stages.
	loss = backend.variable(0.)

	layer_features = layers['block2_conv2']
	content_image_features = layer_features[0, :, :, :]
	combination_features = layer_features[2, :, :, :]

	loss += content_weight * content_loss(content_image_features, combination_features)

	feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

	for layer_name in feature_layers:
		layer_features = layers[layer_name]
		style_features = layer_features[1, :, :, :]
		combination_features = layer_features[2, :, :, :]
		sl = style_loss(style_features, combination_features)
		loss += (style_weight / len(feature_layers)) * sl

	loss += total_variation_weight * total_variation_loss(combination_image)
	# Define needed gradients and solve the optimisation problem
	grads = backend.gradients(loss, combination_image)
	outputs = [loss]
	outputs += grads
	f_outputs = backend.function([combination_image], outputs)

	evaluator = Evaluator(f_outputs)
	x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

	for i in range(iterations):
		print('Start of iteration', i)
		start_time = time.time()
		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
		print('Current loss value:', min_val)
		end_time = time.time()
		print('Iteration %d completed in %ds' % (i, end_time - start_time))

	image = get_image(x, True)
	image.show()

	content_image_name = content_image_path.split('/')[-1]
	style_image_name = style_image_path.split('/')[-1]
	image_name = content_image_name.split('.')[0]+'_in_'+style_image_name.split('.')[0]+'_style.jpg'
	image.save(content_image_path[:-len(content_image_name)]+image_name)

	return 0


def run_on_video():

	content_images, style_images = load_video()
	content_images = get_video_images(content_images)
	style_images = get_video_images(style_images)

	# Channels as the last dimension, using backend Tensorflow
	combination_images = []
	for t in range(len(content_images)):
		combine_image = backend.placeholder((1, height, width, 3))
		combination_images.append(combine_image)

	all_images = []
	for content_image in content_images:
		all_images.append(content_image)
	for style_image in style_images:
		all_images.append(style_image)
	for combine_image in combination_images:
		all_images.append(combine_image)


	input_tensor = backend.concatenate(all_images, axis=0)
	model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
	layers = dict([(layer.name, layer.output) for layer in model.layers])

	losses = []
	for t in range(len(content_images)):
		loss = backend.variable(0.)
		losses.append(loss)

	layer_features = layers['block2_conv2']
	feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

	for content_idx in range(len(content_images)):
		for layer_name in feature_layers:
			layer_features = layers[layer_name]
			for style_img_idx in range(len(style_images)):
				style_features = layer_features[len(content_images) + style_img_idx, :, :, :]
				combination_features = layer_features[len(content_images) + len(style_images) + content_idx, :, :, :]
				style_l = style_loss(style_features, combination_features)
				losses[content_idx] += (style_weight / (len(feature_layers)*len(style_images))) * style_l

	for content_idx in range(len(content_images)):
		losses[content_idx] += total_variation_weight * total_variation_loss(combination_images[content_idx])

	# calculate the gradients
	grads = backend.gradients(losses, combination_images)

	outputs = losses
	outputs += grads
	# Create the function from input combination_img to the loss and gradients
	f_outputs = backend.function(combination_images, outputs)

	evaluator = Evaluator(f_outputs)

	xs = []
	for idx in range(len(content_images)):
		x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.0
		xs.append(x)

	video = cv2.VideoWriter('video-out.avi',-1,1,(width,height))

	for i in range(iterations):
		print('Start of iteration', i)
		start_time = time.time()
		xs, min_val, info = fmin_l_bfgs_b(evaluator.loss, xs, fprime=evaluator.grads, maxfun=20)
		print('Current loss value:', min_val)
		end_time = time.time()
		print('Iteration %d completed in %ds' % (i, end_time - start_time))

		x1 = copy.deepcopy(xs)
		x1 = x1.reshape((len(content_images), 1, height, width, 3))
		for idx in range(len(content_images)):
			x2 = x1[idx]
			x2 = x2.reshape((height, width, 3))
			image = get_image(x2, False)
			video.write(image)
			if i == iterations - 1:
				img_final = Image.fromarray(image)
				img_final.save('result' + str(i) + str(idx) + '.jpg')

	video.release()

	return 0


def main():

	# run_style_transfer()
	run_on_video()

	return 0


if __name__ == '__main__':
	main()

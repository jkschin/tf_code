from tfrecords_main import *
import convert_svhn
import convert_kaggle_mnist
import cv2
import os
import numpy as np
import random

'''
	Input: Format must be in [num_examples, height, width, depth]
'''
def resize_image_array(images, new_size):
	num_examples, height, width, depth = images.shape
	new_image_array = np.empty([num_examples, new_size, new_size, depth])
	for i in xrange(num_examples):
		img = cv2.resize(images[i], (new_size, new_size))
		new_image_array[i] = img
	return new_image_array

'''
	Assume that images and labels are the same, I mean you're retarded if they aren't.
'''
def shuffle_images_and_labels(images, labels, shuffle_count):
	num_examples = images.shape[0]
	for i in xrange(shuffle_count):
		a, b = random.randint(0,num_examples-1),random.randint(0,num_examples-1)
		img_copy = images[a]
		images[a] = images[b]
		images[b] = img_copy
		label_copy = labels[a]
		labels[a] = labels[b]
		labels[b] = label_copy

print ("Converting Kaggle MNIST Data")
k_images, k_labels, k_num_examples = convert_kaggle_mnist.parse_train((os.path.abspath('../data/kaggle_mnist/raw')), 'train.csv') 
k_images = resize_image_array(k_images, 32)
print ("Kaggle Data Shape:", k_images.shape)

print ("Converting SVHN Data")
s_images, s_labels, s_num_examples = convert_svhn.parse((os.path.abspath('../data/svhn/raw')), 'train_32x32.mat') 

print ("SVHN Data Shape:", s_images.shape)
print (k_labels)
print (s_labels)

images = np.concatenate((s_images,k_images))
images = images.astype(np.uint8)
labels = np.concatenate((s_labels,k_labels))
labels = labels.astype(np.uint64)
num_examples = images.shape[0]

shuffle_count = 1000000
print ("Shuffling Data for: ", shuffle_count)
print ("Combined Data:", images.shape, images.dtype)
print ("Combined Labels:", labels.shape, labels.dtype)
shuffle_images_and_labels(images, labels, shuffle_count)
convert_to(images, labels, "kaggle_svhn_combined", num_examples)






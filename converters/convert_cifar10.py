from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

CWD = os.getcwd()


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
	num_examples = images.shape[0]
	rows = images.shape[1]
	cols = images.shape[2]
	depth = images.shape[3]
	if num_examples!=10000:
		raise ValueError("Examples don't match")

	filename = os.path.join(CWD, name + '.tfrecords')
	print('Writing', filename)
	print('Parameters', rows,cols,depth)
	writer = tf.python_io.TFRecordWriter(filename)
	for i in range(num_examples):
		image_raw = images[i].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(rows),
			'width': _int64_feature(cols),
			'depth': _int64_feature(depth),
			'label': _int64_feature(int(labels[i])),
			'image_raw': _bytes_feature(image_raw)}))
		writer.write(example.SerializeToString())

def unpickle():
	#Output images should be identical.
	import random
	def test(img_array1, img_array2, labels):
		rand = random.randint(0,50000)
		im1 = Image.fromarray(img_array1[rand].reshape([3,1024]).T.reshape([32,32,3]))
		im2 = Image.fromarray(img_array2[rand])
		im1.save("im1.jpeg")
		im2.save("im2.jpeg")
		f = open('out.txt','w')
		f.write(str(labels[rand]))

	# NUM_EXAMPLES = 50000
	# CHANNELS = 3
	# IMAGE_SIZE = 32
	# images = [0] * NUM_EXAMPLES
	# labels = [0] * NUM_EXAMPLES
	# for i in range(1,6):
	# 	f = "cifar-10-batches-py/data_batch_" + str(i)
	# 	fo = open(f, 'rb')
	# 	dic = cPickle.load(fo)
	# 	fo.close()
	# 	images[i*10000-10000:i*10000] = dic['data']
	# 	labels[i*10000-10000:i*10000] = dic['labels']
	NUM_EXAMPLES = 10000
	CHANNELS = 3
	IMAGE_SIZE = 32
	images = [0] * NUM_EXAMPLES
	labels = [0] * NUM_EXAMPLES
	f = 'test_batch'
	fo = open(f, 'rb')
	dic = cPickle.load(fo)
	fo.close()
	images[0:10000] = dic['data']
	labels[0:10000] = dic['labels']
	images = np.array(images)
	labels = np.array(labels)
	images_o = np.transpose(images.reshape([NUM_EXAMPLES,CHANNELS,IMAGE_SIZE*IMAGE_SIZE]),(0,2,1)).reshape([NUM_EXAMPLES,IMAGE_SIZE,IMAGE_SIZE,CHANNELS])
	# test(images,images_o,labels)
	return images_o,labels

images,labels = unpickle()
convert_to(images,labels,'test')


# print (len(dic['data']))


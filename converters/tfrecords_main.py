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

'''
images: array of images in the format [example:width:height:depth]
labels: array of labels
'''
def convert_to(images, labels, name, n):
	num_examples = images.shape[0]
	rows = images.shape[1]
	cols = images.shape[2]
	depth = images.shape[3]
	if num_examples!=n:
		raise ValueError("Number of examples don't match")

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
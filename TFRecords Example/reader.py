import tensorflow as tf
import numpy as np
import time
from PIL import Image

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
			serialized_example,
			# Defaults are not specified since both keys are required.
			features={
					'height':tf.FixedLenFeature([], tf.int64),
					'image_raw': tf.FixedLenFeature([], tf.string),
					'label': tf.FixedLenFeature([], tf.int64)
			})
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image = tf.reshape(image,[478, 717, 3])
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	return image

	
'''
Pointers:	Remember to run init_op
			tf.reshape may not be the ideal way.
'''
def run():
	with tf.Graph().as_default():
		with tf.Session() as sess:
			filename_queue = tf.train.string_input_producer(["sample.tfrecords"], num_epochs=1)
			images = read_and_decode(filename_queue)
			image_shape = tf.shape(images)
			init_op = tf.global_variables_initializer()
			sess.run(tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			img = np.array(sess.run(images))
			print img.shape
			print img
			print "Successful read"
			coord.request_stop()
			coord.join(threads)
run()

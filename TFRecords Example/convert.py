# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

CWD = os.getcwd()


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
	rows = images.shape[0]
	cols = images.shape[1]
	depth = images.shape[2]

	filename = os.path.join(CWD, name + '.tfrecords')
	print('Writing', filename)
	writer = tf.python_io.TFRecordWriter(filename)
	image_raw = images.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		'height': _int64_feature(rows),
		'width': _int64_feature(cols),
		'depth': _int64_feature(depth),
		'label': _int64_feature(labels),
		'image_raw': _bytes_feature(image_raw)}))
	writer.write(example.SerializeToString())

img = np.array(Image.open("sample.jpg"))
convert_to(img, 1, 'sample')

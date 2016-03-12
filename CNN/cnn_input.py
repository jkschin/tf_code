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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from constants import *

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64)
      })
  height = tf.cast(features['height'],tf.int32)
  width = tf.cast(features['width'],tf.int32)
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  # image = tf.reshape(image, tf.pack([height, width, 3]))
  # image = tf.reshape(image, [28, 28, 3])
  image = tf.cast(image,tf.float32)
  label = tf.cast(features['label'], tf.int32)
  return image, label

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  filename = os.path.join(data_dir, TRAIN_FILE)
  filename_queue = tf.train.string_input_producer([filename])
  image, label = read_and_decode(filename_queue)
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  distorted_image = tf.random_crop(image, [height, width, 3])
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
  filename = os.path.join(data_dir, TEST_FILE)
  filename_queue = tf.train.string_input_producer([filename])
  image, label = read_and_decode(filename_queue)
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  print ("THIS",image.get_shape)
  
  resized_image = tf.image.resize_images(image, height, width)
  print (resized_image.get_shape)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                           min_fraction_of_examples_in_queue)

  images, label_batch = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=1,
      capacity=min_queue_examples + 3 * batch_size)

  tf.image_summary('images', images)
  return images, tf.reshape(label_batch, [batch_size])

# Author: Samuel Chin

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

'''
Reads a TFRecord file and parses the following information. It is 
important to ensure that the TFRecord has the following "dictionary keys".

If your images vary in size, the proper way to dynamically reshape it is to use
image = tf.reshape(image, tf.pack([height, width, 3]))

However, there are many subtleties in doing that as functions like 
tf.image.resize_image_with_crop_or_pad, requires that the shape be known before hand.
The TF Team is currently working on such a fix, so we shouldn't trouble ourselves with
trying to do this dynamic thing. I raised this on StackOverflow:

http://stackoverflow.com/questions/35773898/how-can-i-use-values-read-from-tfrecords-as-arguments-to-tf-set-shape

With this in mind, the function does only this:

Returns: image, label, height, width, depth, Tensors, all decoded from the TFRecords.
'''
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
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  return image, label, height, width, depth

'''
Inputs:
  image: one image tensor
  label: one label tensor
  min_queue_examples: number of examples to maintain the queue
  batch_size: size of batch generated
Outputs:
  A batch of images and labels

'''
def generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, train):
  if TRAIN or BATCH_EVAL:
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images, max_images=100)
  return images, tf.reshape(label_batch, [batch_size])

def distortions(image):
  distorted_image = tf.random_crop(image, [NETWORK_IMAGE_SIZE, NETWORK_IMAGE_SIZE, 3])
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  return distorted_image


def inputs():
  if TRAIN:
    filename = os.path.join(DATA_DIR, TRAIN_FILE)
  else:
    filename = os.path.join(DATA_DIR, TEST_FILE)
  filename_queue = tf.train.string_input_producer([filename])
  image, label, height, width, depth = read_and_decode(filename_queue)
  image = tf.reshape(image, tf.pack([height, width, 3]))
  # image = tf.reshape(image, [INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
  image = tf.cast(image, tf.float32)

  if TRAIN:
    image = distortions(image)
  else:
    '''
    This is the really retarded part of TensorFlow where the method below
    requires knowing the static shape. I need a fix ASAP.
    '''
    print ("No Distortions")
    image.set_shape([INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, NETWORK_IMAGE_SIZE, NETWORK_IMAGE_SIZE)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, BATCH_SIZE, TRAIN)
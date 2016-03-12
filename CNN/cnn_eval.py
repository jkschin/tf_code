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

"""Evaluation for CIFAR-10.

Accuracy:
cnn_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cnn_eval.py.

Speed:
On a single Tesla K40, cnn_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import os

import cnn
from constants import *

def eval_once(saver, summary_writer, top_k_op, labels, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.h
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cnn_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      # num_iter = int(math.ceil(NUM_EXAMPLES / BATCH_SIZE))
      num_iter = 26032
      true_count = 0  # Counts the number of correct predictions.
      # total_sample_count = num_iter * BATCH_SIZE
      step = 0
      while step < num_iter and not coord.should_stop():
        prediction, label = sess.run([top_k_op[1],labels])
        if prediction==label:
          true_count+=1
        print (step)
        step += 1

      # Compute precision @ 1.
      precision = true_count / num_iter
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    images, labels = cnn.inputs(eval_data=EVAL_DATA)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cnn.inference(images)

    # Calculate predictions.
    # top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_op = tf.nn.top_k(logits, k=1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cnn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(EVAL_DIR,
                                            graph_def=graph_def)

    while True:
      eval_once(saver, summary_writer, top_k_op, labels, summary_op)
      if RUN_ONCE:
        break
      time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):  # pylint: disable=unused-argument
  """Check if tfrecords exists"""
  filename = os.path.join(DATA_DIR, TEST_FILE)
  if not tf.gfile.Exists(filename):
    raise ValueError('Failed to find file: ' + filename)
  print ("Test file: ", TEST_FILE)
  if tf.gfile.Exists(EVAL_DIR):
    tf.gfile.DeleteRecursively(EVAL_DIR)
  tf.gfile.MakeDirs(EVAL_DIR)
  evaluate()


if __name__ == '__main__':
  tf.app.run()

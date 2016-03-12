# Author: Samuel Chin
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

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import combined
import inputs
from constants import *

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    images, labels = inputs.inputs()
    logits = combined.inference(images)
    loss = combined.loss(logits, labels)
    train_op = combined.train(loss, global_step)
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=LOG_DEVICE_PLACEMENT))
    saver = tf.train.Saver(tf.all_variables())

    if tf.gfile.Exists(TRAIN_DIR):
      ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
      last_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      ckpt_dir = os.path.join(CHECKPOINT_DIR,"model.ckpt-" + last_step)
      if ckpt and ckpt_dir:
        tf.gfile.DeleteRecursively(TRAIN_DIR)
        saver.restore(sess, ckpt_dir)
        assign_op = global_step.assign(int(last_step))
        sess.run(assign_op)
        print ("Read old model from: ", ckpt_dir)
        print ("Starting training at: ", sess.run(global_step))        
      else:
        tf.gfile.DeleteRecursively(TRAIN_DIR)
        sess.run(init)
        print ("No model found. Starting training at: ",sess.run(global_step))
    else:
      tf.gfile.MakeDirs(TRAIN_DIR)
      sess.run(init)
      print ("No folder found. Starting training at: ",sess.run(global_step))
    print ("Writing train results to: ", TRAIN_DIR)
    print ("Train file: ", TRAIN_FILE)
    print ("Using Network: ", NETWORK)
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(TRAIN_DIR,
                                            graph_def=sess.graph_def)

    for step in xrange(sess.run(global_step), MAX_STEPS):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = BATCH_SIZE
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 10 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == MAX_STEPS:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  """Check if tfrecords exists"""
  filename = os.path.join(DATA_DIR, TRAIN_FILE)
  if not tf.gfile.Exists(filename):
    raise ValueError('Failed to find file: ' + filename)
  print ("Starting Train")
  train()

if __name__ == '__main__':
  tf.app.run()

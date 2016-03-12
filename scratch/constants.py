import os 
'''
Available Networks:
	CIFAR10
'''


DATASET = "svhn"
NETWORK = "CIFAR10"

DATA_DIR = os.path.abspath(os.path.join('../data', DATASET))
TRAIN_DIR = os.path.abspath(os.path.join('../data',DATASET,'train'))
EVAL_DIR = os.path.abspath(os.path.join('../data',DATASET,'eval'))
CHECKPOINT_DIR = os.path.abspath(os.path.join('../data',DATASET,'train_3000'))
TRAIN_FILE = DATASET + '_train.tfrecords'
TEST_FILE = DATASET + '_test.tfrecords'
# TEST_FILE = 'kaggle_mnist_test.tfrecords'

BATCH_SIZE = 128
TRAIN = True 
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_EXAMPLES = 10000

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

MAX_STEPS = 1000000
LOG_DEVICE_PLACEMENT = False

EVAL_INTERVAL_SECS = 300
RUN_ONCE = True

#placeholder
EVAL_DATA = 'test'
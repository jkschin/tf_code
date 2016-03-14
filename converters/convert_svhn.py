import scipy.io as sio
import os
from tfrecords_main import *
from PIL import Image

def parse(DATA_DIR, filename):
	mat_file = sio.loadmat(os.path.join(DATA_DIR,filename))
	X = mat_file['X']
	y = mat_file['y']
	width, height, depth, num_examples = X.shape
	images = X.reshape([width * height * depth, num_examples]).T.reshape([num_examples, width, height, depth])
	labels = y.reshape(num_examples)
	return images, labels-1, num_examples

# images, labels, num_examples = parse((os.path.abspath('../data/svhn/raw')), 'train_32x32.mat') 
# print images.shape
# convert_to(images,labels-1,"svhn_test",num_examples)



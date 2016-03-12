import scipy.io as sio
import os
from tfrecords_main import *
from PIL import Image

DATA_DIR = "/home/sep/Desktop/svhn"
filename = "test_32x32.mat"

def parse(filename):
	mat_file = sio.loadmat(os.path.join(DATA_DIR,filename))
	X = mat_file['X']
	y = mat_file['y']
	width, height, depth, num_examples = X.shape
	images = X.reshape([width * height * depth, num_examples]).T.reshape([num_examples, width, height, depth])
	labels = y.reshape(num_examples)
	return images, labels, num_examples

images, labels, num_examples = parse(filename)
print images.shape
convert_to(images,labels-1,"svhn_test",num_examples)



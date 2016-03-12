import scipy.io as sio
import os
from tfrecords_main import *
from PIL import Image
import csv

DATA_DIR = "/home/sep/Downloads"
filename = "train.csv"

def parse_train(filename):
	csv = np.genfromtxt(os.path.join(DATA_DIR,filename), delimiter=',')[1:]
	csv = np.array(csv, np.uint8)
	csv = csv.T
	labels = csv[0]
	csv = csv[1:]
	csv = csv.T
	num_examples = len(csv)
	csv = csv.flatten()
	csv = np.vstack((csv,csv,csv))
	csv = csv.T
	images = np.reshape(csv,[num_examples,28,28,3])
	return images, labels, num_examples

def parse_test(filename):
	csv = np.genfromtxt(os.path.join(DATA_DIR,filename), delimiter=',')[1:]
	csv = np.array(csv, np.uint8)
	num_examples = len(csv)
	csv = csv.flatten()
	csv = np.vstack((csv,csv,csv))
	csv = csv.T
	images = np.reshape(csv,[num_examples,28,28,3])
	labels = np.empty([num_examples])
	labels.fill(-1)
	return images, labels, num_examples


# images, labels, num_examples = parse_train("train.csv")
# print (images.shape)
# convert_to(images,labels,"kaggle_mnist_train",num_examples)

images, labels, num_examples = parse_test("test.csv")
print (images.shape)
convert_to(images, labels, "kaggle_mnist_test", num_examples)


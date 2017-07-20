import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Loads MNIST data from the path.
       Returns images, a num_samples by num_features numpy array, and labels, the corresponding label of each sample.
       Source: Sebastian Raska, Python Machine Learning
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        #read in file protocol and num of items from buffer w/big-endianness (MSB first)
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        #the images are 28x28 pixels -> unrolled into a 1d vector
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    images = images
    labels = labels
    return images, labels

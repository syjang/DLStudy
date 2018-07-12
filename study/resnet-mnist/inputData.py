import tensorflow as tf
import numpy as np

class inputdata:
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images # Returns np.array
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.test_data = mnist.test.images # Returns np.array
        self.test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        

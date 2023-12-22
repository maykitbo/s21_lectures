import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential



# TF_ENABLE_ONEDNN_OPTS=1

(x_train, y_train), (x_test, y_test) = mnist.load_.data()

x_train = x_train / 255.0
x_test = x_test / 255.0




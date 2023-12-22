import tensorflow as tf
import os

# import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
# from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images.
x_train = x_train / 255.0
x_test = x_test / 255.0


# Function to update the plots
def update_plots(event):
    for ax in axes.flatten():
        random_index = np.random.randint(0, x_train.shape[0])
        ax.imshow(x_train[random_index], cmap='gray')
        ax.set_title(f"Label: {y_train[random_index]}")
        ax.axis('off')
    plt.draw()

# Create a figure and a set of subplots
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
update_plots(None)  # Initial population of subplots

# Add a button for updating the plots
ax_button = plt.axes([0.45, 0.01, 0.1, 0.075])  # Adjust the position and size as needed
button = Button(ax_button, 'Refresh')
button.on_clicked(update_plots)

plt.show()


if __name__ == '__main__':
    print("Hello World")


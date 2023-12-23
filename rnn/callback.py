# import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import tensorflow as tf
import time
import numpy as np
from pynput.keyboard import Key, Listener


class PlotLossCallback(tf.keras.callbacks.Callback):
    def on_press(self, key):
        pass

    def on_release(self, key):
        if key == Key.esc:
            self.model.stop_training = True
            return False
    
    def listen(self):
        self.listener = Listener(
                on_press=self.on_press,
                on_release=self.on_release)
        self.listener.start()
    
    def plot_d(self):
        pl.cla()
        pl.plot(self.train_loss, label='Training loss')
        pl.plot(self.val_loss, label='Validation loss')
        pl.legend()
        pl.xlabel('Epoch')
        pl.ylabel('Loss')
        display.clear_output(wait=True)
        display.display(pl.gcf())

    def __init__(self, batch_coef):
        self.train_loss = []
        self.val_loss = []
        self.batch_coef = batch_coef
        self.listen()

        self.plot_d()
        
    def __del__(self):
        self.listener.stop()
        self.listener.join()

    def on_epoch_end(self, epoch, logs=None):
        self.val_loss.append(logs['val_loss'])
        self.train_loss.append(logs['loss'])
        self.plot_d()

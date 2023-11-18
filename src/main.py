import tensorflow as tf
from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import time

max_features = 10000  # Number of words to consider as features
max_len = 500  # Cut texts after this number of words

BATCH = 128
EPOCH = 6
VAL_SPLIT = 0.2
LR = 0.0005

TRAIN_LOG = 0

batch_coef = 1.0


def GetData():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # # Combine training and test sets to fit tokenizer on the entire dataset
    # all_sequences = x_train + x_test
    # all_texts = [' '.join([str(i) for i in sequence]) for sequence in all_sequences]
    
    # # Create a tokenizer and fit it on all texts
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(all_texts)

    # # Calculate the number of unique words in the vocabulary
    # num_unique_words = len(tokenizer.word_index)

    # # Set max_features to a percentage of the unique words
    # max_features = int(0.8 * num_unique_words)
    
    # print(max_features)

    # x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    # x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    
    return (x_train, y_train), (x_test, y_test)


class SaveDataCallback(tf.keras.callbacks.Callback):
    counter = 0

    def __init__(self, lstm_size, dropout, dense):
        self.filename = f"../log/training_log_{TRAIN_LOG}_{lstm_size}_{dropout}_{dense}_{SaveDataCallback.counter}.txt"
        self.file = open(self.filename, 'w')  # Open the file for writing
        self.timer = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        # Save data from one epoch to the file
        self.file.write(f"\nEpoch: {epoch}, Training Loss: {logs['loss']}, Training Accuracy: {logs['accuracy']}, Validation Loss: {logs['val_loss']}, Validation Accuracy: {logs['val_accuracy']}\n")
    
    def on_train_end(self, logs=None):
        # Save data from training and close the file
        self.file.write("\nTraining finished.\n")
        self.file.write(f"Final Training Loss: {logs['loss']}, Final Training Accuracy: {logs['accuracy']}, Final Validation Loss: {logs['val_loss']}, Final Validation Accuracy: {logs['val_accuracy']}\n")
        self.file.write(f"Time taken: {time.time() - self.timer}\n")
        self.file.close()
        
        SaveDataCallback.counter += 1  # Increment counter for the next run
    
    def on_batch_end(self, epoch, logs=None):
        # Save data from one epoch to the file
        self.file.write(f"{logs['loss']}")

class PlotLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.batch_x = []
        self.batch_e = 0
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_loss, label='Training loss')
        plt.plot(self.val_loss, label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.ion()  # Turn on interactive mode

    def on_epoch_end(self, epoch, logs=None):
        self.val_loss.append(logs['val_loss'])
        plt.cla()
        plt.plot(self.batch_x, self.train_loss, label='Training loss')
        plt.plot(self.val_loss, label='Validation loss')
        plt.legend()
        plt.pause(0.1)  # Pause to allow the plot to update
        # plt.show()
    
    def on_batch_end(self, batch, logs=None):
        if len(self.val_loss) == 0:
            self.val_loss.append(logs['loss'])
        self.train_loss.append(logs['loss'])
        self.batch_x.append(self.batch_e)
        plt.cla()
        plt.plot(self.batch_x, self.train_loss, label='Training loss')
        plt.plot(self.val_loss, label='Validation loss')
        plt.legend()
        plt.pause(0.01)  # Pause to allow the plot to update
        self.batch_e += batch_coef

def PrintData(x_train, y_train, x_test, y_test):
    # Reverse the word index to get words from indices
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # Decode the reviews
    def decode_review(sequence):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])

    # Print some reviews to the terminal
    num_reviews_to_print = 5
    for i in range(num_reviews_to_print):
        review_index = np.random.randint(0, len(x_train))
        print(f"Review {i + 1}:\n")
        print(decode_review(x_train[review_index]))
        print("\nSentiment (0: Negative, 1: Positive):", y_train[review_index])
        print("\n" + "=" * 50 + "\n")
    
    print(len(x_train))

def lr_schedule(epoch):
    print("LR shedule: ", LR * 0.75 ** epoch)
    return LR * 0.75 ** epoch

def CreateModelType_1():
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(layers.Embedding(max_features, 32))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model

def CreateModelType_2(lstm_size=32, dropout=0.5, dense=1):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, lstm_size, input_length=max_len))
    model.add(Bidirectional(LSTM(lstm_size)))
    model.add(Dropout(dropout))  # Dropout layer for regularization
    model.add(Dense(dense, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model
    
def CreateModelType_3(lstm_size=32, dropout=0.5, dense=1):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, lstm_size, input_length=max_len))
    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(LSTM(lstm_size))
    model.add(Dense(dense, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model   

def CreateModelType_4():
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, 32, input_length=max_len))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(LSTM(32))
    model.add(GlobalMaxPooling1D())  # Global max pooling layer
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model  

def Train(model, x_train, y_train, x_test, y_test, CallBack):
    # Pad sequences to the same length
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    model.fit(x_train, y_train,
            epochs=EPOCH,
            batch_size=BATCH,
            validation_split=VAL_SPLIT,
            callbacks=[CallBack, LearningRateScheduler(lr_schedule)])

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

if __name__ == "__main__":
    # PrintData()
    
    for lstm_s in [32, 16, 26, 44]:
        for drop in [0.3, 0.5, 0.6, 0.8]:
            (x_train, y_train), (x_test, y_test) = GetData()
            batch_coef = BATCH / (len(x_train) * (1 - VAL_SPLIT))
            model = CreateModelType_2(lstm_s, drop, 1)
            Train(model, x_train, y_train, x_test, y_test, SaveDataCallback(lstm_s, drop, 1))
    
    # model = CreateModelType_3()
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback)
    
    # plt.ioff()  # Turn off interactive mode after the loop
    # plt.show()



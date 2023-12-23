import tensorflow as tf
from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.layers import Bidirectional, LSTM, GRU, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

from callback import SaveDataCallback, PlotLossCallback, LayerOutputCallback


max_features = 10000  # Number of words to consider as features
max_len = 500  # Cut texts after this number of words

BATCH = 128
EPOCH = 6
VAL_SPLIT = 0.2
LR = 0.0006
LR_K = 0.9

# TRAIN_LOG = 0

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
    return LR * LR_K ** epoch

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

def CreateModelType_2(lstm_size=32, dropout=0.5, dense=1, batch_norm=True):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, lstm_size, input_length=max_len))
    model.add(Bidirectional(LSTM(lstm_size)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(dense, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

def CreateModelType_3_GRU(gru_size=32, dropout=0.5, dense=1):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, gru_size, input_length=max_len))
    model.add(GRU(gru_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(gru_size))
    model.add(Dropout(dropout))
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
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_size))
    model.add(Dropout(dropout))
    model.add(Dense(dense, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model   

def CreateModelType_4(lstm_size=32, dropout=0.5, dense=1, batch_norm=True):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, lstm_size, input_length=max_len))
    model.add(LSTM(lstm_size, return_sequences=True))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_size, return_sequences=True))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(LSTM(int(lstm_size / 2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

def CreateModelType_5(lstm_size=32, dropout=0.5, dense=1, batch_norm=True):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(max_features, lstm_size, input_length=max_len))
    model.add(LSTM(lstm_size, return_sequences=True))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(GRU(lstm_size, return_sequences=True))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(LSTM(int(lstm_size / 2), return_sequences=True))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(GRU(int(lstm_size / 2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model  

def CreateModelType_6(lstm_size=32, dropout=0.5, dense=1, batch_norm=True):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(Embedding(
                    input_dim=max_features,
                    output_dim=lstm_size,
                    input_length=max_len
                ))
    model.add(Bidirectional(LSTM(
                    units=lstm_size,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    activation='tanh',
                    recurrent_activation='sigmoid'
                )))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dense(
                    units=dense,
                    activation='sigmoid'
                ))

    model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
    return model

def Train(model, x_train, y_train, x_test, y_test, CallBack=None):
    # print("TRAIN BEFORE: ", x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    # print("TRAIN AFTER: ", x_train)

    history = model.fit(x_train, y_train,
            epochs=EPOCH,
            batch_size=BATCH,
            validation_split=VAL_SPLIT,
            callbacks=LearningRateScheduler(lr_schedule))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    return history

if __name__ == "__main__":
    # PrintData()
    
    # for lstm_s in [12, 16, 26, 32]:
    #     for drop in [0.3, 0.5, 0.6, 0.8]:
    #         (x_train, y_train), (x_test, y_test) = GetData()
    #         batch_coef = BATCH / (len(x_train) * (1 - VAL_SPLIT))
    #         model = CreateModelType_3(lstm_s, drop, 1)
    #         Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{lstm_s}_drop_{drop}_x2"))
    
    (x_train, y_train), (x_test, y_test) = GetData()
    batch_coef = BATCH / (len(x_train) * (1 - VAL_SPLIT))
    
    # model = CreateModelType_4(32, 0.6, 1)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{32}_drop_{0.6}_type_4_true"))
    
    # model = CreateModelType_4(32, 0.6, 1, False)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{32}_drop_{0.6}_type_4_false"))
    
    # model = CreateModelType_5(32, 0.6, 1)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{32}_drop_{0.6}_type_5_true"))
    
    # model = CreateModelType_5(32, 0.6, 1, False)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{32}_drop_{0.6}_type_5_false"))
    
    # model = CreateModelType_5(24, 0.6, 1)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{24}_drop_{0.6}_type_5_true"))
    
    # model = CreateModelType_5(24, 0.6, 1, False)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{24}_drop_{0.6}_type_5_false"))
    
    # model = CreateModelType_2(32, 0.6, 1)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{32}_drop_{0.6}_type_2_true"))
    
    # model = CreateModelType_2(32, 0.6, 1, False)
    # Train(model, x_train, y_train, x_test, y_test, SaveDataCallback("imdb", f"lstm_{32}_drop_{0.6}_type_2_false"))
    
    model = CreateModelType_6(32, 0.6, 1, False)
    history = Train(model, x_train, y_train, x_test, y_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    
    # plt.ioff()  # Turn off interactive mode after the loop
    # plt.show()



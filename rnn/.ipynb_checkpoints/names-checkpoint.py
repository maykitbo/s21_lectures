import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.layers import Bidirectional, LSTM, GRU, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import LearningRateScheduler


BATCH = 128
EPOCH = 7
VAL_SPLIT = 0.2
LR = 0.0005

start_token = " "

with open("names") as f:
    names = f.read()[:-1].split('\n')
    names = [start_token + line for line in names]
    
tokens = set()
for name in names:
    tokens.update(set(name))

tokens = list(tokens)

num_tokens = len(tokens)
print (num_tokens, tokens)

tokens.append('<') # <SOS>
tokens.append('>') # <EOS>
tokens.append('_') # <PAD>

token_to_id = {token: idx for idx, token in enumerate(tokens)}

print(token_to_id)

def to_matrix(names, max_len=None, pad=token_to_id['_'], dtype='int32', batch_first=True):
    """Casts a list of names into rnn-digestable matrix"""

    max_len = max_len or max(map(len, names))
    max_len += 1
    names_ix = np.zeros([len(names), max_len], dtype) + pad
    names_ix[:, 0] = token_to_id['<'] # <SOS>

    for i in range(len(names)):
        line_ix = [token_to_id[c] for c in names[i]]
        names_ix[i, 1:len(line_ix)] = line_ix[1:]
        names_ix[i, len(line_ix)] = token_to_id['>'] # <EOS>

    if not batch_first: # convert [batch, time] into [time, batch]
        names_ix = np.transpose(names_ix)

    return names_ix


print('\n'.join(names[::2000]))
print(to_matrix(names[::2000]))


def CreateModelType_1(lstm_size=32, dropout=0.5, batch_norm=True):
    optimizer = Adam(learning_rate=LR)
    model = models.Sequential()
    model.add(layers.Embedding(num_tokens, lstm_size))
    model.add(layers.LSTM(lstm_size))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_tokens, activation='softmax'))
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


# Convert names to matrices
data_matrix = to_matrix(names)

# Split the data into training and validation sets
split_idx = int(len(data_matrix) * (1 - VAL_SPLIT))
train_data, val_data = data_matrix[:split_idx], data_matrix[split_idx:]

# Define your target (output) values. Assuming you want to predict the next token in the sequence.
target = np.eye(num_tokens)[train_data[:, 1:]]  # Assuming one-hot encoding

# Create the model
model = CreateModelType_1()

# Train the model
history = model.fit(train_data[:, :-1], target, batch_size=BATCH, epochs=EPOCH, validation_split=VAL_SPLIT)


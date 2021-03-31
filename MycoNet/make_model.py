# build model

import random as rn
import numpy as np
from tensorflow.keras.models import Sequential
# from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy

def make_model(input_dim, embedding_dim, output_layer, dropout, activation, learning_rate, LSTM_dim):
    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = embedding_dim, name='embedding_layer'))
    # model.add(Bidirectional(LSTM(LSTM_dim, return_sequences=True)))
    # model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(LSTM_dim, activation="tanh")))
    model.add(Dropout(dropout))
    #model.add(Dense(LSTM_dim))
    #model.add(ReLU(activation))
    #model.add(Dropout(dropout))
    model.add(Dense(output_layer, activation='softmax', name="output_layer"))
    adam = Adam(learning_rate)
    model.compile(optimizer=adam, loss = sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model

def split_train_test(X, labels, test_size):
    random_idx = rn.sample(range(len(X)), len(X))
    limit = int(len(X)*(1-test_size))
    random_train = random_idx[0:limit]
    random_test = random_idx[limit:]
    return X[random_train,:],X[random_test,:],labels[random_train],labels[random_test]


import sys, os, time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tsa.utils as utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class lstm_model():

    def __init__(self):
        self.model = None
        self.history = None
        self.X = None
        self.y = None

    def preprocessing(self, df, n_hours = 3, n_features = 8):

        values = df.values

        # integer encode direction
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])

        # ensure all data is float
        values = values.astype('float32')

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # frame as supervised learning
        reframed = utils.series_to_supervised(scaled, n_hours, 1)


        return self

    def fit(self, X, y, X_val=None, y_val=None):

        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        # fit network
        history = model.fit(X, y, epochs=50, batch_size=72,
                            #validation_data=(test_X, test_y),
                            verbose=2,shuffle=False)

        # plot history
        plt.plot(history.history['loss'], label='train')
        #plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()


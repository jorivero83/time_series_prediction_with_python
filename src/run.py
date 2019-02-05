import sys, os, time
import pandas as pd
import numpy as np

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def main():

    # load dataset
    dataset = read_csv('data/pollution.csv', header=0, index_col=0)
    print(dataset.info())

if __name__ == "__main__":

    main()
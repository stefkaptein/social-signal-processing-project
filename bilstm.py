import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input, TimeDistributed
from source.model.load_data import train_test_split, train_test_split_LSTM

from source.model.scoring_metrics import get_windiff, get_pk, get_k_kappa

# these are hyperparameters. 
# TODO: We should try to find good values for thos
batch_size = 64
n_timesteps = 100
train_ratio = 0.4 # ratio of meetings the model is trained on
LSTM_units = 20

datasets = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029 Bns001 Bns002""".split(" ")
results_merged_path = "./results_merged_fixedf0/"

#load the data
X_train_df, Y_train_df, X_test_df, Y_test_df= train_test_split_LSTM(datasets, results_merged_path, n_timesteps, split=train_ratio)

# convert data from dataframes into numpy arrays of in a 3-D shape (samples, timesteps, features)
X_train = X_train_df.values.astype('float32').reshape(-1, n_timesteps, 5)
X_test = X_test_df.values.astype('float32').reshape(-1, n_timesteps, 5)
Y_train = Y_train_df.values.astype('float32').reshape(-1, n_timesteps, 1)
Y_test = Y_test_df.values.astype('float32').reshape(-1, n_timesteps, 1)

# build the model
model = Sequential()
model.add(Bidirectional(LSTM(LSTM_units, return_sequences=True), input_shape=(n_timesteps, 5)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
history=model.fit(X_train, Y_train,
           batch_size=batch_size,
           epochs=100,
           validation_data=[X_test, Y_test])

#measure the performance
predictions = model(X_test).numpy()

#TODO: some weird error appears here
windiff = get_windiff(Y_test, predictions)
pk = get_pk(Y_test, predictions, k=58)
k_k = get_k_kappa(Y_test, predictions)
print("windiff:", windiff)
print("pk:", pk)
print("K_k ", k_k)

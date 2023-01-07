import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input, TimeDistributed
from load_data import train_test_split, train_test_split_LSTM

from scoring_metrics import get_windiff, get_pk

# I just guessed this number; 16, 32, 128 might also work
batch_size = 64

datasets = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029 Bns001 Bns002""".split(" ")
results_merged_path = "./results_merged_fixedf0/"

n_timesteps = 100

#load the data
X_train_df, Y_train_df, X_test_df, Y_test_df= train_test_split_LSTM(datasets, results_merged_path, n_timesteps)

print(X_train_df.shape)
print(Y_train_df.shape)


#TODO: these dataframes need to be converted to tensorflow to pass them to a keras model
X_train = X_train_df.values.astype('float32').reshape(-1, n_timesteps, 5)
X_test = X_test_df.values.astype('float32').reshape(-1, n_timesteps, 5)
Y_train = Y_train_df.values.astype('float32').reshape(-1, n_timesteps, 1)
Y_test = Y_test_df.values.astype('float32').reshape(-1, n_timesteps, 1)

# build the model
# inputs = Input(shape=(None,), dtype="int32")
# model = Sequential()
# model = layers.Input(shape=(None,), dtype="int32")
# model.add(Input(5, 128, input_length=5))
# model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape = ()))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 5)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, Y_train,
           batch_size=batch_size,
           epochs=100,
           validation_data=[X_test, Y_test])

predictions = model(X_test).numpy()
windiff = get_windiff(predictions, Y_test)
pk = get_pk(predictions, Y_test)
print("windiff:", windiff)
print("pk:", pk)

predictions = np.ndarray(Y_test.shape)
for i in range(X_test.shape[0]):
    pred = model(X_test[i].reshape(1, n_timesteps, 5))
    predictions[i] = pred.numpy()

windiff = get_windiff(predictions, Y_test)
pk = get_pk(predictions, Y_test)
print("windiff:", windiff)
print("pk:", pk)
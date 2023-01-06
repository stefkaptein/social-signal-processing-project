import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input
from load_data import train_test_split
from keras import layers

# I just guessed this number; 16, 32, 128 might also work
batch_size = 64

datasets = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029 Bns001 Bns002""".split(" ")
results_merged_path = "./results_merged_fixedf0/"

#load the data
X_train_df, Y__train_df, X_test_df, Y__test_df= train_test_split(datasets, results_merged_path)

#TODO: these dataframes need to be converted to tensorflow to pass them to a keras model
X_train = X_train_df.values.astype('float32')
X_test = X_test_df.values.astype('float32')
Y_train = Y__train_df.values.astype('float32')
Y__test = Y__test_df.values.astype('float32')

# build the model
inputs = Input(shape=(None,), dtype="int32")
model = Sequential()
# model = layers.Input(shape=(None,), dtype="int32")
# model.add(Input(5, 128, input_length=5))
model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape = ()))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

history=model.fit(X_train, Y_train,
           batch_size=batch_size,
           epochs=3,
           validation_data=[X_test, Y__test])
print(history.history['loss'])
print(history.history['accuracy']) 
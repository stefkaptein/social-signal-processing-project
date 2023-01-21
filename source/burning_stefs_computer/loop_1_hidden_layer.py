import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input, TimeDistributed
from source.model.load_data import train_test_split, train_test_split_LSTM

from source.model.scoring_metrics import get_windiff, get_pk, get_k_kappa

from source.model_trainer_and_tester import read_in_dataset_lstm, test_set_evaluate_multiple_lstm
from tensorflow import keras

batch_size = 64

# I optimize on this, I think?
LSTM_units = 20

features = ['pause', 'speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff', 'f0_stds_means']

shifts = [-2, -1, 0, 1, 2]

n_timesteps = len(shifts)
feature_count = len(features)

location = '../../results_merged_f0_stds_fixed/'

X_train, Y_train = read_in_dataset_lstm(features, shifts, to_read='train', location= location)

sample_weight = np.ones(shape=(len(Y_train),))
# I'm gonna increase the weight by the inverse of the proportion of weird examples that there are
# How I define if there is a weird sample is by summing along the 2D squares to find where there's a 1, and then does
# a sum of times there's a 1
# I'm going to do n_timesteps times the inverse count frequency, because in the final version we only predict with the
# center value. So to correct for this I add this increase
new_weight = 5*n_timesteps*len(Y_train)/np.sum(Y_train, axis=1).sum()

# Have to do a flatten() inside because of weird numpy stuff with a length 1 dimension
sample_weight[(np.sum(Y_train, axis=1) >= 1).flatten()] = new_weight
num_layers = 1

results_dict = {}

for hidden_units in range(16, 513, 31):
    model = Sequential()
    # For the input number of units, I'll assume that number of timesteps * features is a good enough value
    for _ in range(num_layers):
        model.add(Bidirectional(LSTM(hidden_units, activation='tanh', return_sequences=True, dropout=0.3),
                                input_shape=(n_timesteps, feature_count)))

    model.add(Bidirectional(LSTM(hidden_units, activation='sigmoid', return_sequences=True, dropout=0.3)))
    # This last time distributed is super important, it follows the output structure of the paper I've been following
    # closely
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    import tensorflow_addons as tfa
    model.compile(loss='binary_crossentropy', optimizer='RMSprop',
                  metrics=[keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
                  weighted_metrics=[]
                  )

    # train the model
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=10,
                        # class_weight= {0:1, 1:10},
                        # sample_weight_mode='temporal',
                        sample_weight=sample_weight,
                        validation_split=0.1,
                        )

    temp_results = test_set_evaluate_multiple_lstm(model, features, shifts, k=58, location=location)

    results_dict[(num_layers, hidden_units)] = pd.concat([temp_results.mean().add_suffix('_mean'), temp_results.std().add_suffix('_std')])

results_df = pd.DataFrame(results_dict)

results_df.to_csv('results_' + str(num_layers) + '.csv')

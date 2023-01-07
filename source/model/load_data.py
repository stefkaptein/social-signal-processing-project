import numpy as np
import pandas as pd
import os
import random


# load a the training and test data
# Input: 
#   datasets: array with dataset names
#   path: path where datasets are stored
#   split: percentage of samples we train on
# Output: a train test split dataset
def train_test_split(datasets, dataset_path, split = 0.4):
    num_meetings = int(len(datasets) * split)

    # Pick num_elements elements randomly
    selected_meetings = random.sample(datasets, num_meetings)

    train_df = pd.DataFrame()
    for elem in selected_meetings:
        path = (os.path.realpath(os.path.join(os.getcwd(), (f"{dataset_path}"+ elem + ".csv"))))
        df = pd.read_csv(path, sep=';')
        train_df = pd.concat([train_df,df], ignore_index=True)

    train_df['speakerChange'] = train_df["speakerChange"].astype(float)
    train_df['boundary'] = train_df["boundary"].astype(float)
    train_df.fillna(0,inplace=True)

    #X_df = train_df[['StartTimeA','EndTimeA','StartTimeB','EndTimeB', 'f0_stds', 'pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
    # I think these are the 5 most useful ones
    X_train_df = train_df[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
    Y_train_df = train_df['boundary']

    test_df = pd.DataFrame()
    for elem in list(set(datasets) - set(selected_meetings)):
        path = (os.path.realpath(os.path.join(os.getcwd(), (f"{dataset_path}"+ elem + ".csv"))))
        df = pd.read_csv(path, sep=';')
        test_df = pd.concat([train_df,df], ignore_index=True)

    test_df['speakerChange'] = test_df["speakerChange"].astype(float)
    test_df['boundary'] = test_df["boundary"].astype(float)
    test_df.fillna(0,inplace=True)

    #X_df = train_df[['StartTimeA','EndTimeA','StartTimeB','EndTimeB', 'f0_stds', 'pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
    # I think these are the 5 most useful ones
    X_test_df = test_df[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
    Y__test_df = test_df['boundary']
    
    return X_train_df, Y_train_df, X_test_df, Y__test_df

# here I split the values into batches of size "timesteps" so we can feed those batches to an LSTM
def train_test_split_LSTM(datasets, dataset_path, timesteps, split = 0.4):
    num_meetings = int(len(datasets) * 0.4)

    # Pick num_elements elements randomly
    selected_meetings = random.sample(datasets, num_meetings)

    train_meetings = []
    for elem in selected_meetings:
        path = (os.path.realpath(os.path.join(os.getcwd(), (f"{dataset_path}"+ elem + ".csv"))))
        df = pd.read_csv(path, sep=';')
        df['speakerChange'] = df["speakerChange"].astype(float)
        df['boundary'] = df["boundary"].astype(float)
        df.fillna(0,inplace=True)
        train_meetings.append(df)
        # train_df = pd.concat([train_df,df], ignore_index=True)

    X_train_df = pd.DataFrame()
    Y_train_df = pd.DataFrame()
    for tm in train_meetings:
        i = 0
        meeting_df = pd.DataFrame()
        while i <= tm.shape[0]-100:
            batch = tm.loc[i+1:i+100]
            x_batch = batch[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
            y_batch = batch['boundary']
            X_train_df = pd.concat([X_train_df,x_batch], ignore_index=True)
            Y_train_df = pd.concat([Y_train_df,y_batch], ignore_index=True)
            i += 100
        # this implementation causes us to throw away the last values
        # TODO: implement a workaround to predict all values


    test_meetings = []
    for elem in list(set(datasets) - set(selected_meetings)):
        path = (os.path.realpath(os.path.join(os.getcwd(), (f"{dataset_path}"+ elem + ".csv"))))
        df = pd.read_csv(path, sep=';')
        df['speakerChange'] = df["speakerChange"].astype(float)
        df['boundary'] = df["boundary"].astype(float)
        df.fillna(0,inplace=True)
        test_meetings.append(df)


    X_test_df = pd.DataFrame()
    Y_test_df = pd.DataFrame()

    for tm in test_meetings:
        i = 0
        meeting_df = pd.DataFrame()
        while i <= tm.shape[0]-100:
            batch = tm.loc[i+1:i+100]
            x_batch = batch[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
            y_batch = batch['boundary']
            X_test_df = pd.concat([X_test_df,x_batch], ignore_index=True)
            Y_test_df = pd.concat([Y_test_df,y_batch], ignore_index=True)
            i += 100

    return X_train_df, Y_train_df, X_test_df, Y_test_df

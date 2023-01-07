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
    Y_test_df = test_df['boundary']

    return X_train_df, Y_train_df, X_test_df, Y_test_df
import numpy as np
import pandas as pd
import os
import random

def filter_lvl(df,highest_lvl):
    df['boundary1']=None

    curr_lvl=df.at[0,'Level']
    for i, row_i in df.iterrows():
        lvl = row_i['Level']

        df.at[i,'boundary1']=0

        if lvl>curr_lvl and curr_lvl<highest_lvl:
            df.at[i-1,'boundary1']=1
            curr_lvl=lvl
        
        if lvl<curr_lvl and curr_lvl<highest_lvl:
            df.at[i-1,'boundary1']=1
            curr_lvl=lvl

    
    print(df['boundary1'].unique())

    return df


# load a the training and test data
# Input: 
#   datasets: array with dataset names
#   path: path where datasets are stored
#   split: percentage of samples we train on
# Output: a train test split dataset
def train_test_split(datasets, dataset_path, test_split = 0.4):
    train_num_meetings = int(len(datasets) * (1-test_split))

    # Pick num_elements elements randomly
    train_selected_meetings = random.sample(datasets, train_num_meetings)

    train_df = pd.DataFrame()
    for elem in train_selected_meetings:
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
    for elem in list(set(datasets) - set(train_selected_meetings)):
        path = (os.path.realpath(os.path.join(os.getcwd(), (f"{dataset_path}"+ elem + ".csv"))))
        df = pd.read_csv(path, sep=';')       
        test_df = pd.concat([test_df,df], ignore_index=True)

    test_df['speakerChange'] = test_df["speakerChange"].astype(float)
    test_df['boundary'] = test_df["boundary"].astype(float)
    test_df.fillna(0,inplace=True)

    #X_df = train_df[['StartTimeA','EndTimeA','StartTimeB','EndTimeB', 'f0_stds', 'pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
    # I think these are the 5 most useful ones
    X_test_df = test_df[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
    Y__test_df = test_df['boundary']

    return X_train_df, Y_train_df, X_test_df, Y__test_df

# in general sames as above
# however here I split the values into batches of size "timesteps" so we can feed those batches to an LSTM
# thus the returned X_train_df has a shape of e.g. (192, timesteps, 5)

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
        while i <= tm.shape[0]-timesteps:
            batch = tm.loc[i:i+timesteps-1]
            if batch.shape[0] != timesteps:
                print(batch.shape[0], i, i+timesteps, tm.shape[0])
            x_batch = batch[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
            y_batch = batch['boundary']
            X_train_df = pd.concat([X_train_df,x_batch], ignore_index=True)
            Y_train_df = pd.concat([Y_train_df,y_batch], ignore_index=True)
            i += timesteps
        # this implementation causes us to throw away the last values e.g. if we have 1567 rows and timesteps=100 we throw away the last 67
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
        while i <= tm.shape[0]-timesteps:
            batch = tm.loc[i:i+timesteps-1]
            if batch.shape[0] != timesteps:
                st = batch.shape[0]
                print(batch.shape[0], i, i+timesteps, tm.shape[0])
            x_batch = batch[['pause','speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']]
            y_batch = batch['boundary']
            X_test_df = pd.concat([X_test_df,x_batch], ignore_index=True)
            Y_test_df = pd.concat([Y_test_df,y_batch], ignore_index=True)
            i += timesteps

    return X_train_df, Y_train_df, X_test_df, Y_test_df

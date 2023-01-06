import os

import pandas as pd

columns = ['segID', 'similarity', 'StartTimeA', 'EndTimeA', 'StartTimeB', 'EndTimeB', "f0_means", "f0_stds", "pause"]


def combine_and_normalize_features(folder_path: str):
    # walk directory
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            feature_df = pd.read_csv(file_path, sep=';', usecols=columns)
            feature_df["similarity"] = feature_df["similarity"]
            yes = 1

if __name__ == "__main__":
    combine_and_normalize_features("../../results")
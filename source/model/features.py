import os

import pandas as pd

# columns = ['segID', 'similarity', 'StartTimeA', 'EndTimeA', 'StartTimeB', 'EndTimeB', "f0_means", "f0_stds", "pause"]
columns = ['segID', 'similarity', 'StartTimeA', 'EndTimeA', 'StartTimeB', 'EndTimeB', "pause"]


def combine_and_normalize_features(folder_path: str):
    res_df = pd.DataFrame(columns=columns)
    # walk directory
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                feature_df = pd.read_csv(file_path, sep=';', usecols=columns)
                feature_df["similarity"] = feature_df["similarity"].str.replace(r"\[|\]", "", regex=True).astype('float')
                # series = feature_df["f0_means"].str[1:-1].str.replace("\'", "").str.split(",", expand=True)
                res_df = pd.concat([res_df, feature_df])
            except ValueError as e:
                try:
                    feature_df = pd.read_csv(file_path, sep=',', usecols=columns)
                    feature_df["similarity"] = feature_df["similarity"].str.replace(r"\[|\]", "", regex=True).astype(
                        'float')
                    res_df = pd.concat([res_df, feature_df])
                except ValueError as e:
                    print(f"Error while reading file {file_path}: {e}")
    res_df.to_csv(f"{folder_path}/all_final_no_audio.csv", sep=',', index=False)


if __name__ == "__main__":
    combine_and_normalize_features("../../results")

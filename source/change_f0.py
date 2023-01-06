import pandas as pd
import ast

from data import meeting_names

meetings = meeting_names()
for meeting in meetings:
    all_features = pd.read_csv(f"../social-signal-processing-project/results_merged/{meeting}.csv", sep=';')
    f0_diffs = []
    f0_diff_to_baselines = []
    for index, row in all_features.iterrows():
        if index != all_features.shape[0]-1:
            f01 = ast.literal_eval(row['f0_means'])
            f02 = ast.literal_eval(all_features.at[index+1, 'f0_means'])

            if len(f01)!=0 and len(f02)!=0:
                f0_diffs.append(float(f01[-1]) - float(f02[0]))
            else:
                f0_diffs.append(0)

        else:
            f0_diffs.append(0)

        if len(f01)!= 0:
            f0_diff_to_baselines.append(float(f01[-1]) - 1)
        else: 
            f0_diff_to_baselines.append(0)      

    all_features.drop(['f0_means'], axis=1, inplace=True)
    all_features['f0_diff'] = f0_diffs
    all_features['f0_baseline_diff'] = f0_diff_to_baselines
    print(all_features.head())
    all_features.to_csv(f"../social-signal-processing-project/results_merged_fixedf0/{meeting}.csv", sep=';', index=False)
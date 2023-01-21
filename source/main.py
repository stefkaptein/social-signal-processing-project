import pandas as pd
import numpy as np

from data import meeting_names
from data_parser import extract_all_data_for_meeting_and_write_to_files
from extract_audio_features import extract_audio_features_and_write_to_file
from text_features import extract_text_features_and_write_to_file


def create_feature_vector(meeting_name: str) -> pd.DataFrame:
    text_features = pd.read_csv(f"../out/{meeting_name}_text_features_of_segments.csv", sep=';')
    audio_features = pd.read_csv(f"../out/{meeting_name}_audio_features_of_segments.csv", sep=';')
    final_df = audio_features.merge(text_features, left_on='segID', right_on='id', how='inner')
    final_df.to_csv(f"../results/{meeting_name}_final.csv", sep=';', index=False)
    return final_df


def add_lvl_info(meeting_name):
    boundaries_df = pd.read_csv(f"../out/{meeting_name}_topic_segments.csv", sep=';')
    final_df = pd.read_csv(f"../results_merged_fixedf0/{meeting_name}.csv", sep=';')
    final_df['Level'] = None

    for i, row_i in final_df.iterrows():
        id = int((row_i['segID'][15:]).replace(",",""))
        for j, row_j in boundaries_df.iterrows():
            seg_a_id = int((row_j['First Segment id'][15:]).replace(",",""))
            seg_b_id = int((row_j['Last Segment id'][15:]).replace(",",""))
            if (id>=seg_a_id) and (id<=seg_b_id):
                new_lvl = int(row_j['Level'])
            final_df.at[i,'Level'] = new_lvl

    final_df.to_csv(f"../results_merged_fixedfo_lvl/{meeting_name}.csv", sep=';', index=False)


if __name__ == "__main__":
    oui = True
    meetings = meeting_names()
    features = pd.DataFrame()
    for meeting in meetings:
        print(meeting)
        if meeting == 'Bns003':
            oui = False
        
        add_lvl_info(meeting)
            
        # extract_all_data_for_meeting_and_write_to_files(meeting,oui)
        # extract_audio_features_and_write_to_file(meeting)
        # extract_text_features_and_write_to_file(meeting)
        # meeting_feature_vector = create_feature_vector(meeting)
        # pd.concat([features, meeting_feature_vector])
    # features.to_csv("../social-signal-processing-project/results_merged_fixedfo_lvl/all_final.csv", sep=';', index=False)

import pandas as pd

from data import meeting_names
from data_parser import extract_all_data_for_meeting_and_write_to_files
from extract_audio_features import extract_audio_features_and_write_to_file
from text_features import extract_text_features_and_write_to_file

out_path = "../out"
result_path = "../results"


def create_feature_vector(meeting_name: str) -> pd.DataFrame:
    text_features = pd.read_csv(f"{out_path}/{meeting_name}_text_features_of_segments.csv", sep=';')
    audio_features = pd.read_csv(f"{out_path}/{meeting_name}_audio_features_of_segments.csv", sep=';')
    final_df = audio_features.merge(text_features, left_on='segID', right_on='id', how='inner')
    final_df.to_csv(f"{out_path}/{meeting_name}_final.csv", sep=';', index=False)
    return final_df


if __name__ == "__main__":
    meetings = meeting_names()
    features = pd.DataFrame()
    for meeting in meetings:
        extract_all_data_for_meeting_and_write_to_files(meeting)
        extract_audio_features_and_write_to_file(meeting)
        extract_text_features_and_write_to_file(meeting)
        meeting_feature_vector = create_feature_vector(meeting)
        meeting_feature_vector.to_csv(f"{result_path}/{meeting}_final.csv")
        pd.concat([features, meeting_feature_vector])
    features.to_csv(f"{result_path}/all_final.csv", sep=';', index=False)

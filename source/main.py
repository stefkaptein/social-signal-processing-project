import pandas as pd

from source.text_features import get_sentence_similarity


def create_feature_vector():
    df = pd.read_csv("../out/Bed002_segments_final_sorted.csv", sep=';')
    text_features = get_sentence_similarity(df, "Text")
    audio_features = pd.read_csv("../out/Bed002_audio_features_of_segments.csv", sep=';')
    return audio_features.merge(text_features, left_on='segID', right_on='id', how='inner')


if __name__ == "__main__":
    feature_vector = create_feature_vector()

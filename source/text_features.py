from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd


def clean_dataframe_text(df_to_clean: pd.DataFrame, text_col_name: str):
    df_to_clean[text_col_name] = df_to_clean[text_col_name].str.replace(r"\[.*?\]", "", regex=True)
    df_to_clean[text_col_name] = df_to_clean[text_col_name].str.strip()
    df_to_clean.drop(df_to_clean[df_to_clean[text_col_name] == ""].index, inplace=True)


def get_sentence_similarity(dataframe: pd.DataFrame, text_col_name: str) -> pd.DataFrame:
    dataframe_copy = dataframe.copy()

    clean_dataframe_text(dataframe_copy, text_col_name)
    text_col = dataframe_copy[text_col_name].values.tolist()
    id_col = dataframe_copy["id"].values.tolist()[:-1]

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    encodings = model.encode(text_col)
    similarity_on_boundary = []
    for i in range(len(encodings) - 1):
        similarity_on_boundary.append(cosine_similarity(encodings[i].reshape(1, -1), encodings[i + 1].reshape(1, -1)))
    res_df = pd.DataFrame({"id": id_col, "similarity": similarity_on_boundary})
    return res_df


def extract_text_features_and_write_to_file(meeting_name: str):
    segments_df = pd.read_csv(f"../social-signal-processing-project/out/{meeting_name}_segments_final_sorted.csv", sep=';')
    res_df = get_sentence_similarity(segments_df, "Text")
    res_df.to_csv(f"../social-signal-processing-project/out/{meeting_name}_text_features_of_segments.csv", sep=';', index=False)

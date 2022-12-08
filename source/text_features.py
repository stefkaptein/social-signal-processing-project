from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd


def get_sentence_similarity(dataframe: pd.DataFrame, text_col_name: str):
    text_col = dataframe[text_col_name]

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    encodings = model.encode(text_col)
    similarity_on_boundary = []
    for i in range(len(encodings) - 1):
        similarity_on_boundary.append(cosine_similarity(encodings[i].reshape(1, -1), encodings[i + 1].reshape(1, -1)))
    print(similarity_on_boundary)


if __name__ == "__main__":
    df = pd.read_csv("../out/Bdb001_segments_final_sorted", sep=';')
    get_sentence_similarity(df, "Text")

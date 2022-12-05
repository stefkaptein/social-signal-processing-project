from typing import List
import config as cfg
import os 
import pandas as pd

from transcript import read_full_transcript_phrase,read_full_transcript_word

def get_phrases_df(meeting_name):
    path = (os.path.realpath(os.path.join(os.getcwd(), ("source\ICSI_original_transcripts\\text\\transcripts\\"+meeting_name+".mrt"))))
    df_phrases = read_full_transcript_phrase(path)
    return df_phrases

def get_words_df(meeting_name):
    main_path = (os.path.realpath(os.path.join(os.getcwd(), ("source\ICSI_original_transcripts\\audio\ICSIplus\Words\\"))))
    all_files_names = [f for f in os.listdir(main_path) if os.path.isfile(os.path.join(main_path, f))]

    df_words = []
    for path in (path for path in all_files_names if meeting_name in path):
        meeting_path = (main_path+"\\"+path)
        participant = path[7]
        df_words.append(read_full_transcript_word(meeting_path,participant))
    
    df_whole_words = pd.concat(df_words)
    df_whole_words = df_whole_words.reset_index(drop=True)
    return df_whole_words


if __name__ == "__main__":
    meeting_name = "Bdb001"

    # df_phrases = get_phrases_df(meeting_name)
    # df_phrases.to_csv(("out\\"+meeting_name+"_phrases"), sep='\t')

    df_words = get_words_df(meeting_name)
    df_words.to_csv(("out\\"+meeting_name+"_words"), sep='\t')

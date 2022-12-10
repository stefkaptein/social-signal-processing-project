from typing import List
import config as cfg
import os 
import pandas as pd
import copy
import numpy as np

from transcript import read_full_transcript_phrase,read_full_transcript_word,read_full_transcript_segment,read_full_transcript_topic_segments, read_full_transcript_prosody

meeting_name = "Bed002"


def get_phrases_df(meeting_name):
    path = (os.path.realpath(os.path.join(os.getcwd(), ("data\ICSI_original_transcripts\\transcripts\\"+meeting_name+".mrt"))))
    df_phrases = read_full_transcript_phrase(path)
    return df_phrases

def get_words_df(meeting_name):
    main_path = (os.path.realpath(os.path.join(os.getcwd(), ("data\ICSI_core_NXT\ICSI\Words\\"))))
    all_files_names = [f for f in os.listdir(main_path) if os.path.isfile(os.path.join(main_path, f))]

    df_words = []
    for path in (path for path in all_files_names if meeting_name in path):
        meeting_path = (main_path+"\\"+path)
        participant = path[7]
        df_words.append(read_full_transcript_word(meeting_path,participant))
    
    df_whole_words = pd.concat(df_words)
    df_whole_words = df_whole_words.reset_index(drop=True)
    return df_whole_words

def get_prosodies(meeting_name):
    main_path = (os.path.realpath(os.path.join(os.getcwd(), ("data\ICSIplus\Contributions\AutomaticProsody\\"))))
    all_files_names = [f for f in os.listdir(main_path) if os.path.isfile(os.path.join(main_path, f))]

    df_prosodies = []
    for path in (path for path in all_files_names if meeting_name in path):
        meeting_path = (main_path+"\\"+path)
        participant = path[7]
        df_prosodies.append(read_full_transcript_prosody(meeting_path, participant))
    
    df_whole_prosodies = pd.concat(df_prosodies)
    df_whole_prosodies = df_whole_prosodies.reset_index(drop=True)
    return df_whole_prosodies

def get_segments_df(meeting_name):
    main_path = (os.path.realpath(os.path.join(os.getcwd(), ("data\ICSI_core_NXT\ICSI\Segments\\"))))
    all_files_names = [f for f in os.listdir(main_path) if os.path.isfile(os.path.join(main_path, f))]

    df_segments = []
    for path in (path for path in all_files_names if meeting_name in path):
        meeting_path = (main_path+"\\"+path)
        participant = path[7]
        df_segments.append(read_full_transcript_segment(meeting_path,participant))
    
    df_whole_segments = pd.concat(df_segments)
    df_whole_segments = df_whole_segments.reset_index(drop=True)
    return df_whole_segments

def filter_words(df_words,txt,w1):
    word_id = df_words['id'].loc[df_words['id']==w1].values[0]
    if "w" in word_id:
        return txt
    elif (df_words['Description'].loc[df_words['id']==w1].values[0]) and ("disfmarker" not in word_id) and ("pause" not in word_id):
        txt=('[',df_words['Description'].loc[df_words['id']==w1].values[0],']')
        return"".join(map(str, txt))
    elif "pause" in word_id:
        if (meeting_name+".pause.1" != word_id):
            duration = float(df_words['EndTime'].loc[df_words['id']==w1])-float(df_words['StartTime'].loc[df_words['id']==w1])
            return ("[Pause - "+str(round(duration,4))+"s]")
        return "[Pause]"
    elif "disfmarker" in df_words['id'].loc[df_words['id']==w1].values[0]:
        return "[Diskmarker]"

def filter_prosodies(df_prosodies,w1):
    return df_prosodies['f0_mean'].loc[df_prosodies['words_id']==w1], df_prosodies['f0_std'].loc[df_prosodies['words_id']==w1]

def combine_df(df_words,df_segments,df_prosodies):
    df_final = copy.deepcopy(df_segments)
    df_final['Text'] = None
    df_final['f0_means'] = None
    df_final['f0_stds'] = None
    for i,row in df_final.iterrows():
        if i%100==0: print(i)
        words_id = (row["words_id"]).split("#")
        words = words_id[1].split("..")
        w1 = words[0]
        if len(words)==1:
            txt = df_words['Text'].loc[df_words['id']==w1[3:len(w1)-1]].values[0]
            
            df_final['Text'].loc[i]=filter_words(df_words,txt,w1[3:len(w1)-1])
            f0_means, f0_stds = filter_prosodies(df_prosodies,w1[3:len(w1)-1])
            df_final['f0_means'].loc[i]= f0_means
            df_final['f0_stds'].loc[i]= f0_stds
        else:
            w2 = words[1]
            word_inter = get_all_words(df_words,w1,w2,df_final['Participant2'].loc[i])
            f0_means, f0_stds = get_all_prosodies_for_words(df_prosodies, w1, w2, df_final['Participant2'].loc[i], df_words)
            df_final['Text'].loc[i]=word_inter
            df_final['f0_means'].loc[i] = f0_means
            df_final['f0_stds'].loc[i] = f0_stds

    # return df_final 
    return df_final.drop("words_id",axis=1)

def get_all_words(df_words,w1,w2,participant):
    #order df_words by start time
    index_w1 = df_words[df_words['id']==w1[3:len(w1)-1]].index.values[0]
    index_w2 = df_words[df_words['id']==w2[3:len(w2)-1]].index.values[0]

    words=[]
    for i in range(index_w1,index_w2+1):
        if df_words['Participant'].iloc[i]==participant:
            txt = df_words['Text'].iloc[i]
            filtered_txt = filter_words(df_words,txt,df_words['id'].iloc[i])
            words.append(filtered_txt)
    
    return " ".join(map(str, words))

def filter_words(df_words,txt,w1):
    word_id = df_words['id'].loc[df_words['id']==w1].values[0]
    if "w" in word_id:
        return txt
    elif (df_words['Description'].loc[df_words['id']==w1].values[0]) and ("disfmarker" not in word_id) and ("pause" not in word_id):
        txt=('[',df_words['Description'].loc[df_words['id']==w1].values[0],']')
        return"".join(map(str, txt))
    elif "pause" in word_id:
        if (meeting_name+".pause.1" != word_id):
            duration = float(df_words['EndTime'].loc[df_words['id']==w1])-float(df_words['StartTime'].loc[df_words['id']==w1])
            return ("[Pause - "+str(round(duration,4))+"s]")
        return "[Pause]"
    elif "disfmarker" in df_words['id'].loc[df_words['id']==w1].values[0]:
        return "[Diskmarker]"

def get_all_prosodies_for_words(df_prosodies,w1,w2,participant, df_words):
    meeting, w, index_w1 = w1[3:len(w1)-1].split('.')
    meeting, w, index_w2 = w2[3:len(w2)-1].split('.')

    f0_means=[]
    f0_stds=[]
    for i in range(int(index_w1.replace(",", "")),int(index_w2.replace(",", ""))+1):
        prosody = df_prosodies[df_prosodies['words_id']==meeting+".w."+"{:,}".format(i)]
        if not prosody.empty:
            if prosody['Participant'].iloc[0]==participant:
                f0_mean = prosody['f0_mean'].values[0]
                f0_std = prosody['f0_std'].values[0]
                f0_means.append(f0_mean)
                f0_stds.append(f0_std)
    return f0_means, f0_stds


if __name__ == "__main__":
    # df_phrases = get_phrases_df(meeting_name)
    # df_phrases.to_csv(("out\\"+meeting_name+"_phrases"), sep='\t')

    df_words = get_words_df(meeting_name)
    df_words.to_csv(("out\\"+meeting_name+"_words"), sep='\t')

    df_prosodies = get_prosodies(meeting_name)
    df_prosodies.to_csv(("out\\"+meeting_name+"_prosodies"), sep=';')

    df_segments = get_segments_df(meeting_name)
    df_segments.to_csv(("out\\"+meeting_name+"_segments"), sep='\t')

    df_segments_final = combine_df(df_words,df_segments, df_prosodies)
    df_segments_final.to_csv(("out\\"+meeting_name+"_segments_final"), sep=';')

    df_topic_segments = get_topic_segments_df(meeting_name)
    df_topic_segments.to_csv(("out\\"+meeting_name+"_topic_segments"), sep='\t')


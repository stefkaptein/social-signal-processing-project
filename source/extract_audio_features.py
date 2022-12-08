import pandas as pd

# this file will extract the audio features of the meeting specified below into a file
# "out\meeting_name_audio_features_of_segments"
meeting_name = "Bed002"

# explanation of the features: f0_means and f0_std are all the recorded f0_means, f0_std values in a segment. There
# might be none, one or multiple values pause is how much time there is between the end of speaking in segmentA and
# the start of speech in segmentB if this number is positive there was a break. if it is negative the speech
# overlapped speakerchange indicates whether the speaker of segmentA and segmentB are the same speakerchange=1 or not
# speakerchange=0


# loads the segments, sorts them according to starting time and saves them
def sort_segments(path):
    segments = pd.read_csv(path, sep=';')
    segments_sorted = segments.sort_values(by=['StartTime'])
    segments_sorted.to_csv(("out\\"+meeting_name+"_segments_final_sorted"), sep=';')


# Input: path to sorted segments
# iterates throught the adjacent pairs of segements and extracts the audio features
def iterate_through_pairs(path):
    seg = pd.read_csv(path, sep=';')
    
    audio_dat  = pd.DataFrame(columns = ['segID', 'StartTimeA', 'EndTimeA', 'StartTimeB', 'EndTimeB', 'f0_means', 'f0_stds', 'pause', 'speakerChange'])
    for ind in range(len(seg)-1):
        id = seg['id'][ind]
        startA = seg['StartTime'][ind]
        endA = seg['EndTime'][ind]
        startB = seg['StartTime'][ind+1]
        endB = seg['EndTime'][ind+1]
        speakerChange = False
        if seg['Participant1'][ind] != seg['Participant1'][ind+1]:
            speakerChange = True
        pause = seg['StartTime'][ind+1] - seg['EndTime'][ind]
        f0_mean = seg['f0_means'][ind]
        f0_stds = seg['f0_stds'][ind]
        audio_dat = audio_dat.append({
            'segID':id, 
            'StartTimeA':startA, 
            'EndTimeA':endA, 
            'StartTimeB':startB, 
            'EndTimeB':endB, 
            'f0_means':f0_mean, 
            'f0_stds':f0_stds, 
            'pause':pause, 
            'speakerChange':speakerChange}, ignore_index=True)
    audio_dat.to_csv(("out\\"+meeting_name+"_audio_features_of_segments"), sep=';')
        

sort_segments("out\\"+meeting_name+"_segments_final")
iterate_through_pairs("out\\"+meeting_name+"_segments_final_sorted")
# data.to_csv(("out\\"+meeting_name+"_audio_features_of_segments"), sep=';')

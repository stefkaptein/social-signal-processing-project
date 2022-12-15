import pandas as pd


# explanation of the features: f0_means and f0_std are all the recorded f0_means, f0_std values in a segment. There
# might be none, one or multiple values pause is how much time there is between the end of speaking in segmentA and
# the start of speech in segmentB if this number is positive there was a break. if it is negative the speech
# overlapped speakerchange indicates whether the speaker of segmentA and segmentB are the same speakerchange=1 or not
# speakerchange=0

out_path = "../out"


# loads the segments, sorts them according to starting time and saves them
def sort_segments(path, meeting_name):
    segments = pd.read_csv(path, sep=';')
    segments_sorted = segments.sort_values(by=['StartTime'])
    segments_sorted.to_csv(f"{out_path}/{meeting_name}_segments_final_sorted.csv", sep=';')


# Input: path to sorted segments
# iterates throught the adjacent pairs of segements and extracts the audio features
def iterate_through_pairs(path, meeting_name):
    seg = pd.read_csv(path, sep=';')

    audio_dat = pd.DataFrame(
        columns=['segID', 'StartTimeA', 'EndTimeA', 'StartTimeB', 'EndTimeB', 'f0_means', 'f0_stds', 'pause',
                 'speakerChange'])
    for ind in range(len(seg) - 1):
        id = seg['id'][ind]
        startA = seg['StartTime'][ind]
        endA = seg['EndTime'][ind]
        startB = seg['StartTime'][ind + 1]
        endB = seg['EndTime'][ind + 1]
        speakerChange = False
        if seg['Participant1'][ind] != seg['Participant1'][ind + 1]:
            speakerChange = True
        pause = seg['StartTime'][ind + 1] - seg['EndTime'][ind]
        f0_mean = seg['f0_means'][ind]
        f0_stds = seg['f0_stds'][ind]
        audio_dat = pd.concat([
            audio_dat,
            pd.DataFrame([{
                'segID': id,
                'StartTimeA': startA,
                'EndTimeA': endA,
                'StartTimeB': startB,
                'EndTimeB': endB,
                'f0_means': f0_mean,
                'f0_stds': f0_stds,
                'pause': pause,
                'speakerChange': speakerChange}])]
        )
    audio_dat.to_csv(f"{out_path}/{meeting_name}_audio_features_of_segments.csv", sep=';')


def extract_audio_features_and_write_to_file(meeting_name):
    sort_segments(f"{out_path}/{meeting_name}_segments_final.csv", meeting_name)
    iterate_through_pairs(f"{out_path}/{meeting_name}_segments_final_sorted.csv", meeting_name)
# data.to_csv(("out\\"+meeting_name+"_audio_features_of_segments"), sep=';')

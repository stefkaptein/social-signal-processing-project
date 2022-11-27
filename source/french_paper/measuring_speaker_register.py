import librosa
import numpy as np


def get_fundamental_frequncies(file_path):
    """Gets the fundamental frequency (f0) of a given audio file using the librosa library
    implementation. Estimates the quantiles first to get a better idea of what the fmin and fmax
    should be, instead of using the default. Done mainly because the paper itself recommends
    doing this.

    This means the process goes like this: Get the f0 with the initial default values. Calculate
    the q65 and q35 quantiles, and use the formula the paper uses. Then re-run the algorithm for
    the f0 with the new minimium and maximum

    :param file_path: The filepath of the sound file to use
    :return The f0 of the corresponding audiofile given by file_path"""
    y, sr = librosa.load(file_path)
    f0, voiced_flag, voiced_probs = librosa.pyin(y)
    lower_quantile = np.quantile(f0, 0.35)
    upper_quantile = np.quantile(f0, 0.65)

    lower_quantile = lower_quantile*0.72
    upper_quantile = upper_quantile*1.9

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=lower_quantile, fmax=upper_quantile)

    return f0

def get_key_difference(f0_scores, tone_shifts):
    """Method to obtain the key difference between two consecutive units for the key

    So key is defined as the mean for an f0 score. And the key is calculated in hertz.

    The problem for this is that I need to extract the tone shifts from the dataset somehow.
    I'm guessing they'll exist in the large dataset, but not in the smaller one... FML"""


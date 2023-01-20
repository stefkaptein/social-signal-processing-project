import os

import numpy as np
import os
from collections import defaultdict
import scipy.stats
import matplotlib.pyplot as plt
from pylab import *

# OPTIONS:
# Decision Tree (DT) or Naive Bayes (NB)
model = 'DT'
# leave one feature out (loo) or only one feature (oof)
method = 'loo'

def tofloat(list):
    newList = []
    for elem in list:
        newList.append(float(elem))
    return newList

path = (os.path.realpath(os.path.join(os.getcwd(), (f"source\simple_models.txt"))))
file = open(path, 'r')
lines = file.readlines()

combination=['pause', 'speakerChange', 'similarity', 'f0_diff', 'f0_baseline_diff']

# read out the numbers from the text files
# sorry for the spaghetti code
for i in range(len(lines)):

    # this is the baseline with all the features in it
    if str(combination) in lines[i]:
        baseline_DT_windiff = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        baseline_DT_pk= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        baseline_DT_kappa= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        baseline_NB_windiff= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        baseline_NB_pk= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        baseline_NB_kappa= tofloat(lines[i+8][12:-1].strip('][').split(', '))

    #these are the "only one feature"
    elif '[\'pause\']' in lines[i]:
        pause_DT_windiff = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        pause_DT_pk= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        pause_DT_kappa= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        pause_NB_windiff= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        pause_NB_pk= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        pause_NB_kappa= tofloat(lines[i+8][12:-1].strip('][').split(', '))
    elif '[\'similarity\']' in lines[i]:
        similarity_DT_windiff = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        similarity_DT_pk= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        similarity_DT_kappa= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        similarity_NB_windiff= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        similarity_NB_pk= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        similarity_NB_kappa= tofloat(lines[i+8][12:-1].strip('][').split(', '))
    elif '[\'f0_diff\']' in lines[i]:
        f0_diff_DT_windiff = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        f0_diff_DT_pk= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        f0_diff_DT_kappa= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        f0_diff_NB_windiff= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        f0_diff_NB_pk= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        f0_diff_NB_kappa= tofloat(lines[i+8][12:-1].strip('][').split(', '))
    elif '[\'speakerChange\']' in lines[i]:
        speaker_change_DT_windiff = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        speaker_change_DT_pk= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        speaker_change_DT_kappa= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        speaker_change_NB_windiff= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        speaker_change_NB_pk= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        speaker_change_NB_kappa= tofloat(lines[i+8][12:-1].strip('][').split(', '))
    elif '[\'f0_baseline_diff\']' in lines[i]:
        f0_baseline_DT_windiff = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        f0_baseline_DT_pk= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        f0_baseline_DT_kappa= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        f0_baseline_NB_windiff= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        f0_baseline_NB_pk= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        f0_baseline_NB_kappa= tofloat(lines[i+8][12:-1].strip('][').split(', '))

    # here are the leave-one-out runs. Signaled with _loo at the end
    elif not 'pause' in lines[i] and 'similarity' in lines[i] and 'f0_diff' in lines[i] and 'speakerChange' in lines[i] and 'f0_baseline_diff' in lines[i]:
        #all except for pause
        pause_DT_windiff_loo = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        pause_DT_pk_loo= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        pause_DT_kappa_loo= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        pause_NB_windiff_loo= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        pause_NB_pk_loo= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        pause_NB_kappa_loo= tofloat(lines[i+8][12:-1].strip('][').split(', '))

    elif 'pause' in lines[i] and not 'similarity' in lines[i] and 'f0_diff' in lines[i] and 'speakerChange' in lines[i] and 'f0_baseline_diff' in lines[i]:
        #all except for similarity
        similarity_DT_windiff_loo = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        similarity_DT_pk_loo= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        similarity_DT_kappa_loo= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        similarity_NB_windiff_loo= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        similarity_NB_pk_loo= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        similarity_NB_kappa_loo= tofloat(lines[i+8][12:-1].strip('][').split(', '))

    elif 'pause' in lines[i] and 'similarity' in lines[i] and not 'f0_diff' in lines[i] and 'speakerChange' in lines[i] and 'f0_baseline_diff' in lines[i]:
        #all except for f0_diff
        f0_diff_DT_windiff_loo = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        f0_diff_DT_pk_loo= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        f0_diff_DT_kappa_loo= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        f0_diff_NB_windiff_loo= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        f0_diff_NB_pk_loo= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        f0_diff_NB_kappa_loo= tofloat(lines[i+8][12:-1].strip('][').split(', '))

    elif 'pause' in lines[i] and 'similarity' in lines[i] and 'f0_diff' in lines[i] and not 'speakerChange' in lines[i] and 'f0_baseline_diff' in lines[i]:
        #all except for speakerChange
        speaker_change_DT_windiff_loo = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        speaker_change_DT_pk_loo= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        speaker_change_DT_kappa_loo= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        speaker_change_NB_windiff_loo= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        speaker_change_NB_pk_loo= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        speaker_change_NB_kappa_loo= tofloat(lines[i+8][12:-1].strip('][').split(', '))

    elif 'pause' in lines[i] and 'similarity' in lines[i] and 'f0_diff' in lines[i] and 'speakerChange' in lines[i] and not 'f0_baseline_diff' in lines[i]:
        #all except for f0_baseline_diff
        f0_baseline_DT_windiff_loo = tofloat(lines[i+2][12:-1].strip('][').split(', '))
        f0_baseline_DT_pk_loo= tofloat(lines[i+3][7:-1].strip('][').split(', '))
        f0_baseline_DT_kappa_loo= tofloat(lines[i+4][12:-1].strip('][').split(', '))

        f0_baseline_NB_windiff_loo= tofloat(lines[i+6][12:-1].strip('][').split(', '))
        f0_baseline_NB_pk_loo= tofloat(lines[i+7][7:-1].strip('][').split(', '))
        f0_baseline_NB_kappa_loo= tofloat(lines[i+8][12:-1].strip('][').split(', '))


if method == 'loo' and model == 'NB':

    pause_median_windiff = np.median(pause_NB_windiff_loo)
    similarity_median_windiff = np.median(similarity_NB_windiff_loo)
    f0_diff_median_windiff = np.median(f0_diff_NB_windiff_loo)
    speaker_change_median_windiff = np.median(speaker_change_NB_windiff_loo)
    f0_baseline_median_windiff = np.median(f0_baseline_NB_windiff_loo)
    baseline_median_windiff = np.median(baseline_NB_windiff)

    pause_avg_windiff = sum(pause_NB_windiff_loo)/len(pause_NB_windiff_loo)
    similarity_avg_windiff = sum(similarity_NB_windiff_loo)/len(similarity_NB_windiff_loo)
    f0_diff_avg_windiff = sum(f0_diff_NB_windiff_loo)/len(f0_diff_NB_windiff_loo)
    speaker_change_avg_windiff = sum(speaker_change_NB_windiff_loo)/len(speaker_change_NB_windiff_loo)
    f0_baseline_avg_windiff = sum(f0_baseline_NB_windiff_loo)/len(f0_baseline_NB_windiff_loo)
    baseline_avg_windiff = sum(baseline_NB_windiff)/len(baseline_NB_windiff)

    pause_median_pk = np.median(pause_NB_pk_loo)
    similarity_median_pk = np.median(similarity_NB_pk_loo)
    f0_diff_median_pk = np.median(f0_diff_NB_pk_loo)
    speaker_change_median_pk = np.median(speaker_change_NB_pk_loo)
    f0_baseline_median_pk = np.median(f0_baseline_NB_pk_loo)
    baseline_median_pk = np.median(baseline_NB_pk)

    pause_avg_pk = sum(pause_NB_pk_loo)/len(pause_NB_pk_loo)
    similarity_avg_pk = sum(similarity_NB_pk_loo)/len(similarity_NB_pk_loo)
    f0_diff_avg_pk = sum(f0_diff_NB_pk_loo)/len(f0_diff_NB_pk_loo)
    speaker_change_avg_pk = sum(speaker_change_NB_pk_loo)/len(speaker_change_NB_pk_loo)
    f0_baseline_avg_pk = sum(f0_baseline_NB_pk_loo)/len(f0_baseline_NB_pk_loo)
    baseline_avg_pk = sum(baseline_NB_pk)/len(baseline_NB_pk)

    bars_avg_windiff = [baseline_avg_windiff, pause_avg_windiff, similarity_avg_windiff, f0_diff_avg_windiff, speaker_change_avg_windiff, f0_baseline_avg_windiff]
    bars_median_windiff = [baseline_median_windiff, pause_median_windiff, similarity_median_windiff, f0_diff_median_windiff, speaker_change_median_windiff, f0_baseline_median_windiff]
    yerr_lower_normal_windiff = [baseline_avg_windiff - np.quantile(baseline_NB_windiff, 0.25), pause_avg_windiff - np.quantile(pause_NB_windiff_loo, 0.25), similarity_avg_windiff - np.quantile(similarity_NB_windiff_loo, 0.25), f0_diff_avg_windiff - np.quantile(f0_diff_NB_windiff_loo, 0.25), speaker_change_avg_windiff - np.quantile(speaker_change_NB_windiff_loo, 0.25), f0_baseline_avg_windiff - np.quantile(f0_baseline_NB_windiff_loo, 0.25)]
    yerr_upper_normal_windiff = [np.quantile(baseline_NB_windiff, 0.75) - baseline_avg_windiff, np.quantile(pause_NB_windiff_loo, 0.75) - pause_avg_windiff, np.quantile(similarity_NB_windiff_loo, 0.75) - similarity_avg_windiff, np.quantile(f0_diff_NB_windiff_loo, 0.75) - f0_diff_avg_windiff, np.quantile(speaker_change_NB_windiff_loo, 0.75) - speaker_change_avg_windiff, np.quantile(f0_baseline_NB_windiff_loo, 0.75) - f0_baseline_avg_windiff]

    bars_pk = [baseline_avg_pk, pause_avg_pk, similarity_avg_pk, f0_diff_avg_pk, speaker_change_avg_pk, f0_baseline_avg_pk]
    bars_median_pk = [baseline_median_pk, pause_median_pk, similarity_median_pk, f0_diff_median_pk, speaker_change_median_pk, f0_baseline_median_pk]
    yerr_lower_pk = [baseline_avg_pk - np.quantile(baseline_NB_pk, 0.25), pause_avg_pk - np.quantile(pause_NB_pk_loo, 0.25), similarity_avg_pk - np.quantile(similarity_NB_pk_loo, 0.25), f0_diff_avg_pk - np.quantile(f0_diff_NB_pk_loo, 0.25), speaker_change_avg_pk - np.quantile(speaker_change_NB_pk_loo, 0.25), f0_baseline_avg_pk - np.quantile(f0_baseline_NB_pk_loo, 0.25)]
    yerr_upper_pk = [np.quantile(baseline_NB_pk, 0.75) - baseline_avg_pk, np.quantile(pause_NB_pk_loo, 0.75) - pause_avg_pk, np.quantile(similarity_NB_pk_loo, 0.75) - similarity_avg_pk, np.quantile(f0_diff_NB_pk_loo, 0.75) - f0_diff_avg_pk, np.quantile(speaker_change_NB_pk_loo, 0.75) - speaker_change_avg_pk, np.quantile(f0_baseline_NB_pk_loo, 0.75) - f0_baseline_avg_pk]


    x = np.arange(len(bars_avg_windiff))
    ax1 = plt.subplot(1,1,1)
    w = 0.3

    plt.xticks(x + w, ['all\nfeatures', 'no textual\nsimilarity', 'no pause\nduration', 'no speaker\nchange', 'no f0\ndifference', 'no f0\nbaseline'], rotation='horizontal')
    # plt.xticks(x + w, ['only textual\nsimilarity', 'only pause\nduration', 'only speaker\nchange', 'only f0\ndifference', 'only f0\nbaseline'], rotation='horizontal')
    scen = ax1.bar(x, bars_avg_windiff, width=w, color='r', align='center', yerr=[yerr_lower_normal_windiff,yerr_upper_normal_windiff], capsize=4, zorder=1)
    pointscen = ax1.scatter(x, bars_median_windiff, s=25, color='black', zorder=2)
    # ax2 = ax1.twinx()

    aff = ax1.bar(x + w, bars_pk, width=w,color='g',align='center', yerr=[yerr_lower_pk,yerr_upper_pk], capsize=4, zorder=1)
    pointaff = ax1.scatter(x + w, bars_median_pk, s=25, color='black', zorder=2)

    plt.legend([scen, aff],['WinDiff', 'Pk'], loc=3)
    plt.show()

if method == 'loo' and model == 'DT':

    pause_median_windiff = np.median(pause_DT_windiff_loo)
    similarity_median_windiff = np.median(similarity_DT_windiff_loo)
    f0_diff_median_windiff = np.median(f0_diff_DT_windiff_loo)
    speaker_change_median_windiff = np.median(speaker_change_DT_windiff_loo)
    f0_baseline_median_windiff = np.median(f0_baseline_DT_windiff_loo)
    baseline_median_windiff = np.median(baseline_DT_windiff)

    pause_avg_windiff = sum(pause_DT_windiff_loo)/len(pause_DT_windiff_loo)
    similarity_avg_windiff = sum(similarity_DT_windiff_loo)/len(similarity_DT_windiff_loo)
    f0_diff_avg_windiff = sum(f0_diff_DT_windiff_loo)/len(f0_diff_DT_windiff_loo)
    speaker_change_avg_windiff = sum(speaker_change_DT_windiff_loo)/len(speaker_change_DT_windiff_loo)
    f0_baseline_avg_windiff = sum(f0_baseline_DT_windiff_loo)/len(f0_baseline_DT_windiff_loo)
    baseline_avg_windiff = sum(baseline_DT_windiff)/len(baseline_DT_windiff)

    pause_median_pk = np.median(pause_DT_pk_loo)
    similarity_median_pk = np.median(similarity_DT_pk_loo)
    f0_diff_median_pk = np.median(f0_diff_DT_pk_loo)
    speaker_change_median_pk = np.median(speaker_change_DT_pk_loo)
    f0_baseline_median_pk = np.median(f0_baseline_DT_pk_loo)
    baseline_median_pk = np.median(baseline_DT_pk)

    pause_avg_pk = sum(pause_DT_pk_loo)/len(pause_DT_pk_loo)
    similarity_avg_pk = sum(similarity_DT_pk_loo)/len(similarity_DT_pk_loo)
    f0_diff_avg_pk = sum(f0_diff_DT_pk_loo)/len(f0_diff_DT_pk_loo)
    speaker_change_avg_pk = sum(speaker_change_DT_pk_loo)/len(speaker_change_DT_pk_loo)
    f0_baseline_avg_pk = sum(f0_baseline_DT_pk_loo)/len(f0_baseline_DT_pk_loo)
    baseline_avg_pk = sum(baseline_DT_pk)/len(baseline_DT_pk)

    bars_avg_windiff = [baseline_avg_windiff, pause_avg_windiff, similarity_avg_windiff, f0_diff_avg_windiff, speaker_change_avg_windiff, f0_baseline_avg_windiff]
    bars_median_windiff = [baseline_median_windiff, pause_median_windiff, similarity_median_windiff, f0_diff_median_windiff, speaker_change_median_windiff, f0_baseline_median_windiff]
    yerr_lower_normal_windiff = [baseline_avg_windiff - np.quantile(baseline_DT_windiff, 0.25), pause_avg_windiff - np.quantile(pause_DT_windiff_loo, 0.25), similarity_avg_windiff - np.quantile(similarity_DT_windiff_loo, 0.25), f0_diff_avg_windiff - np.quantile(f0_diff_DT_windiff_loo, 0.25), speaker_change_avg_windiff - np.quantile(speaker_change_DT_windiff_loo, 0.25), f0_baseline_avg_windiff - np.quantile(f0_baseline_DT_windiff_loo, 0.25)]
    yerr_upper_normal_windiff = [np.quantile(baseline_DT_windiff, 0.75) - baseline_avg_windiff, np.quantile(pause_DT_windiff_loo, 0.75) - pause_avg_windiff, np.quantile(similarity_DT_windiff_loo, 0.75) - similarity_avg_windiff, np.quantile(f0_diff_DT_windiff_loo, 0.75) - f0_diff_avg_windiff, np.quantile(speaker_change_DT_windiff_loo, 0.75) - speaker_change_avg_windiff, np.quantile(f0_baseline_DT_windiff_loo, 0.75) - f0_baseline_avg_windiff]

    bars_pk = [baseline_avg_pk, pause_avg_pk, similarity_avg_pk, f0_diff_avg_pk, speaker_change_avg_pk, f0_baseline_avg_pk]
    bars_median_pk = [baseline_median_pk, pause_median_pk, similarity_median_pk, f0_diff_median_pk, speaker_change_median_pk, f0_baseline_median_pk]
    yerr_lower_pk = [baseline_avg_pk - np.quantile(baseline_DT_pk, 0.25), pause_avg_pk - np.quantile(pause_DT_pk_loo, 0.25), similarity_avg_pk - np.quantile(similarity_DT_pk_loo, 0.25), f0_diff_avg_pk - np.quantile(f0_diff_DT_pk_loo, 0.25), speaker_change_avg_pk - np.quantile(speaker_change_DT_pk_loo, 0.25), f0_baseline_avg_pk - np.quantile(f0_baseline_DT_pk_loo, 0.25)]
    yerr_upper_pk = [np.quantile(baseline_DT_pk, 0.75) - baseline_avg_pk, np.quantile(pause_DT_pk_loo, 0.75) - pause_avg_pk, np.quantile(similarity_DT_pk_loo, 0.75) - similarity_avg_pk, np.quantile(f0_diff_DT_pk_loo, 0.75) - f0_diff_avg_pk, np.quantile(speaker_change_DT_pk_loo, 0.75) - speaker_change_avg_pk, np.quantile(f0_baseline_DT_pk_loo, 0.75) - f0_baseline_avg_pk]


    x = np.arange(len(bars_avg_windiff))
    ax1 = plt.subplot(1,1,1)
    w = 0.3

    plt.xticks(x + w, ['all\nfeatures', 'no textual\nsimilarity', 'no pause\nduration', 'no speaker\nchange', 'no f0\ndifference', 'no f0\nbaseline'], rotation='horizontal')
    # plt.xticks(x + w, ['only textual\nsimilarity', 'only pause\nduration', 'only speaker\nchange', 'only f0\ndifference', 'only f0\nbaseline'], rotation='horizontal')
    scen = ax1.bar(x, bars_avg_windiff, width=w, color='r', align='center', yerr=[yerr_lower_normal_windiff,yerr_upper_normal_windiff], capsize=4, zorder=1)
    pointscen = ax1.scatter(x, bars_median_windiff, s=25, color='black', zorder=2)
    # ax2 = ax1.twinx()

    aff = ax1.bar(x + w, bars_pk, width=w,color='g',align='center', yerr=[yerr_lower_pk,yerr_upper_pk], capsize=4, zorder=1)
    pointaff = ax1.scatter(x + w, bars_median_pk, s=25, color='black', zorder=2)

    plt.legend([scen, aff],['WinDiff', 'Pk'], loc=3)
    plt.show()

if method == 'oof' and model == 'NB':

    pause_median_windiff = np.median(pause_NB_windiff)
    similarity_median_windiff = np.median(similarity_NB_windiff)
    f0_diff_median_windiff = np.median(f0_diff_NB_windiff)
    speaker_change_median_windiff = np.median(speaker_change_NB_windiff)
    f0_baseline_median_windiff = np.median(f0_baseline_NB_windiff)

    pause_avg_windiff = sum(pause_NB_windiff)/len(pause_NB_windiff)
    similarity_avg_windiff = sum(similarity_NB_windiff)/len(similarity_NB_windiff)
    f0_diff_avg_windiff = sum(f0_diff_NB_windiff)/len(f0_diff_NB_windiff)
    speaker_change_avg_windiff = sum(speaker_change_NB_windiff)/len(speaker_change_NB_windiff)
    f0_baseline_avg_windiff = sum(f0_baseline_NB_windiff)/len(f0_baseline_NB_windiff)

    pause_median_pk = np.median(pause_NB_pk)
    similarity_median_pk = np.median(similarity_NB_pk)
    f0_diff_median_pk = np.median(f0_diff_NB_pk)
    speaker_change_median_pk = np.median(speaker_change_NB_pk)
    f0_baseline_median_pk = np.median(f0_baseline_NB_pk)

    pause_avg_pk = sum(pause_NB_pk)/len(pause_NB_pk)
    similarity_avg_pk = sum(similarity_NB_pk)/len(similarity_NB_pk)
    f0_diff_avg_pk = sum(f0_diff_NB_pk)/len(f0_diff_NB_pk)
    speaker_change_avg_pk = sum(speaker_change_NB_pk)/len(speaker_change_NB_pk)
    f0_baseline_avg_pk = sum(f0_baseline_NB_pk)/len(f0_baseline_NB_pk)

    bars_avg_windiff = [pause_avg_windiff, similarity_avg_windiff, f0_diff_avg_windiff, speaker_change_avg_windiff, f0_baseline_avg_windiff]
    bars_median_windiff = [pause_median_windiff, similarity_median_windiff, f0_diff_median_windiff, speaker_change_median_windiff, f0_baseline_median_windiff]
    yerr_lower_normal_windiff = [pause_avg_windiff - np.quantile(pause_NB_windiff, 0.25), similarity_avg_windiff - np.quantile(similarity_NB_windiff, 0.25), f0_diff_avg_windiff - np.quantile(f0_diff_NB_windiff, 0.25), speaker_change_avg_windiff - np.quantile(speaker_change_NB_windiff, 0.25), f0_baseline_avg_windiff - np.quantile(f0_baseline_NB_windiff, 0.25)]
    yerr_upper_normal_windiff = [np.quantile(pause_NB_windiff, 0.75) - pause_avg_windiff, np.quantile(similarity_NB_windiff, 0.75) - similarity_avg_windiff, np.quantile(f0_diff_NB_windiff, 0.75) - f0_diff_avg_windiff, np.quantile(speaker_change_NB_windiff, 0.75) - speaker_change_avg_windiff, np.quantile(f0_baseline_NB_windiff, 0.75) - f0_baseline_avg_windiff]

    bars_pk = [pause_avg_pk, similarity_avg_pk, f0_diff_avg_pk, speaker_change_avg_pk, f0_baseline_avg_pk]
    bars_median_pk = [pause_median_pk, similarity_median_pk, f0_diff_median_pk, speaker_change_median_pk, f0_baseline_median_pk]
    yerr_lower_pk = [pause_avg_pk - np.quantile(pause_NB_pk, 0.25), similarity_avg_pk - np.quantile(similarity_NB_pk, 0.25), f0_diff_avg_pk - np.quantile(f0_diff_NB_pk, 0.25), speaker_change_avg_pk - np.quantile(speaker_change_NB_pk, 0.25), f0_baseline_avg_pk - np.quantile(f0_baseline_NB_pk, 0.25)]
    yerr_upper_pk = [np.quantile(pause_NB_pk, 0.75) - pause_avg_pk, np.quantile(similarity_NB_pk, 0.75) - similarity_avg_pk, np.quantile(f0_diff_NB_pk, 0.75) - f0_diff_avg_pk, np.quantile(speaker_change_NB_pk, 0.75) - speaker_change_avg_pk, np.quantile(f0_baseline_NB_pk, 0.75) - f0_baseline_avg_pk]


    x = np.arange(len(bars_avg_windiff))
    ax1 = plt.subplot(1,1,1)
    w = 0.3

    # plt.xticks(x + w, ['all\nfeatures', 'no textual\nsimilarity', 'no pause\nduration', 'no speaker\nchange', 'no f0\ndifference', 'no f0\nbaseline'], rotation='horizontal')
    plt.xticks(x + w, ['only textual\nsimilarity', 'only pause\nduration', 'only speaker\nchange', 'only f0\ndifference', 'only f0\nbaseline'], rotation='horizontal')
    scen = ax1.bar(x, bars_avg_windiff, width=w, color='r', align='center', yerr=[yerr_lower_normal_windiff,yerr_upper_normal_windiff], capsize=4, zorder=1)
    pointscen = ax1.scatter(x, bars_median_windiff, s=25, color='black', zorder=2)
    # ax2 = ax1.twinx()

    aff = ax1.bar(x + w, bars_pk, width=w,color='g',align='center', yerr=[yerr_lower_pk,yerr_upper_pk], capsize=4, zorder=1)
    pointaff = ax1.scatter(x + w, bars_median_pk, s=25, color='black', zorder=2)

    plt.legend([scen, aff],['WinDiff', 'Pk'], loc=3)
    plt.show()

if method == 'oof' and model == 'DT':

    pause_median_windiff = np.median(pause_DT_windiff)
    similarity_median_windiff = np.median(similarity_DT_windiff)
    f0_diff_median_windiff = np.median(f0_diff_DT_windiff)
    speaker_change_median_windiff = np.median(speaker_change_DT_windiff)
    f0_baseline_median_windiff = np.median(f0_baseline_DT_windiff)

    pause_avg_windiff = sum(pause_DT_windiff)/len(pause_DT_windiff)
    similarity_avg_windiff = sum(similarity_DT_windiff)/len(similarity_DT_windiff)
    f0_diff_avg_windiff = sum(f0_diff_DT_windiff)/len(f0_diff_DT_windiff)
    speaker_change_avg_windiff = sum(speaker_change_DT_windiff)/len(speaker_change_DT_windiff)
    f0_baseline_avg_windiff = sum(f0_baseline_DT_windiff)/len(f0_baseline_DT_windiff)

    pause_median_pk = np.median(pause_DT_pk)
    similarity_median_pk = np.median(similarity_DT_pk)
    f0_diff_median_pk = np.median(f0_diff_DT_pk)
    speaker_change_median_pk = np.median(speaker_change_DT_pk)
    f0_baseline_median_pk = np.median(f0_baseline_DT_pk)

    pause_avg_pk = sum(pause_DT_pk)/len(pause_DT_pk)
    similarity_avg_pk = sum(similarity_DT_pk)/len(similarity_DT_pk)
    f0_diff_avg_pk = sum(f0_diff_DT_pk)/len(f0_diff_DT_pk)
    speaker_change_avg_pk = sum(speaker_change_DT_pk)/len(speaker_change_DT_pk)
    f0_baseline_avg_pk = sum(f0_baseline_DT_pk)/len(f0_baseline_DT_pk)

    bars_avg_windiff = [pause_avg_windiff, similarity_avg_windiff, f0_diff_avg_windiff, speaker_change_avg_windiff, f0_baseline_avg_windiff]
    bars_median_windiff = [pause_median_windiff, similarity_median_windiff, f0_diff_median_windiff, speaker_change_median_windiff, f0_baseline_median_windiff]
    yerr_lower_normal_windiff = [pause_avg_windiff - np.quantile(pause_DT_windiff, 0.25), similarity_avg_windiff - np.quantile(similarity_DT_windiff, 0.25), f0_diff_avg_windiff - np.quantile(f0_diff_DT_windiff, 0.25), speaker_change_avg_windiff - np.quantile(speaker_change_DT_windiff, 0.25), f0_baseline_avg_windiff - np.quantile(f0_baseline_DT_windiff, 0.25)]
    yerr_upper_normal_windiff = [np.quantile(pause_DT_windiff, 0.75) - pause_avg_windiff, np.quantile(similarity_DT_windiff, 0.75) - similarity_avg_windiff, np.quantile(f0_diff_DT_windiff, 0.75) - f0_diff_avg_windiff, np.quantile(speaker_change_DT_windiff, 0.75) - speaker_change_avg_windiff, np.quantile(f0_baseline_DT_windiff, 0.75) - f0_baseline_avg_windiff]

    bars_pk = [pause_avg_pk, similarity_avg_pk, f0_diff_avg_pk, speaker_change_avg_pk, f0_baseline_avg_pk]
    bars_median_pk = [pause_median_pk, similarity_median_pk, f0_diff_median_pk, speaker_change_median_pk, f0_baseline_median_pk]
    yerr_lower_pk = [pause_avg_pk - np.quantile(pause_DT_pk, 0.25), similarity_avg_pk - np.quantile(similarity_DT_pk, 0.25), f0_diff_avg_pk - np.quantile(f0_diff_DT_pk, 0.25), speaker_change_avg_pk - np.quantile(speaker_change_DT_pk, 0.25), f0_baseline_avg_pk - np.quantile(f0_baseline_DT_pk, 0.25)]
    yerr_upper_pk = [np.quantile(pause_DT_pk, 0.75) - pause_avg_pk, np.quantile(similarity_DT_pk, 0.75) - similarity_avg_pk, np.quantile(f0_diff_DT_pk, 0.75) - f0_diff_avg_pk, np.quantile(speaker_change_DT_pk, 0.75) - speaker_change_avg_pk, np.quantile(f0_baseline_DT_pk, 0.75) - f0_baseline_avg_pk]


    x = np.arange(len(bars_avg_windiff))
    ax1 = plt.subplot(1,1,1)
    w = 0.3

    # plt.xticks(x + w, ['all\nfeatures', 'no textual\nsimilarity', 'no pause\nduration', 'no speaker\nchange', 'no f0\ndifference', 'no f0\nbaseline'], rotation='horizontal')
    plt.xticks(x + w, ['only textual\nsimilarity', 'only pause\nduration', 'only speaker\nchange', 'only f0\ndifference', 'only f0\nbaseline'], rotation='horizontal')
    scen = ax1.bar(x, bars_avg_windiff, width=w, color='r', align='center', yerr=[yerr_lower_normal_windiff,yerr_upper_normal_windiff], capsize=4, zorder=1)
    pointscen = ax1.scatter(x, bars_median_windiff, s=25, color='black', zorder=2)
    # ax2 = ax1.twinx()

    aff = ax1.bar(x + w, bars_pk, width=w,color='g',align='center', yerr=[yerr_lower_pk,yerr_upper_pk], capsize=4, zorder=1)
    pointaff = ax1.scatter(x + w, bars_median_pk, s=25, color='black', zorder=2)

    plt.legend([scen, aff],['WinDiff', 'Pk'], loc=3)
    plt.show()
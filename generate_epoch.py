"""
@author: Yundong Wang, Zimu Li
@reference: UC San Diego COGS 189 Winter 2019 Assignment 1, writen by Alessandro D'Amico
"""
import numpy as np                            # for dealing with data
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir


def generate_epoch(file_path, channels, fs, eeg_filter, stimulus_times = None, baseline = True,  epoch_s = -200, epoch_e = 1300, bl_s = -400, bl_e = -300):
    """
    :description: Generating epoch given csv file. Make sure the csv file layout meets the requirement.
        It should contain 'Time' column that represents timepoints, and the time should start from 0.
        If your csv file does not have FeedBackEvent indicating the stimulus, you must pass stumulus_times.
        Here we used a butter bandpass filter, but you can change to your favorite one.

    :file_path (String): path to your csv file
    :channels ([String]): array of channels to epoch
    :fs (float): sampling rate
    :eeg_filter (function): the filter you want to apply to raw eeg data
    :stimulus_times ([float], optional): The time points that stimulus occur
    :baseline (boolean, optional): whether you want to apply baseline correction after epoching
    :epoch_s (int, optional): epoch starting time relative to stmulus in miliseconds
    :epoch_e (int, optional): epoch ending time relative to stmulus in miliseconds
    :bl_s (int, optional): baseline starting time relative to stmulus in miliseconds
    :bl_e (int, optional): baseline ending time relative to stmulus in miliseconds

    :rtype (3d-nparray): epoched data with dimension (stimulus_per_subj, number_of_channels, number_of_time_points)
    """
    # read dataand data selection
    train_data = pd.read_csv(file_path)

    train_data.loc[:,'Time'] = train_data.loc[:,'Time']*1000
    raw_eeg  = train_data[channels].values.T

    time_df = train_data['Time'].values
    train_data['index'] = train_data.index.values
    if stimulus_times is None:
        mark_indices = np.asarray(train_data[train_data['FeedBackEvent']==1].index).flatten()
    else:
        mark_indices = np.round(np.asarray(stimulus_times).flatten() * fs).astype(int)
        
    # Define the bounds of our epoch as well as our baseline
    b_s = int((abs(epoch_s) + bl_s) * (fs / 1000)) # index in epoch_df where our baseline begins
    b_e = int((abs(epoch_s) + bl_e) * (fs / 1000)) # index in epoch_df where our baseline ends
    # Let's calculate the length our epoch with our given sampling rate
    epoch_len = int((abs(epoch_s) + abs(epoch_e)) * (fs / 1000))

    # Let's define some helpful variables to make our extraction easier
    e_s = int((epoch_s * (fs / 1000))) # effectively the number of indices before marker we want
    e_e = int((epoch_e * (fs / 1000))) # effectively the number of indices after marker we want

    # Epoch the data
    final_epoch = np.empty((mark_indices.shape[0], epoch_len, 0), float)
    for channel in channels:
        epoch = np.zeros(shape = (int(mark_indices.shape[0]), epoch_len))
        raw_eeg = train_data[channel].values

        ################# You may want to apply your own filter ################
        clean_eeg = eeg_filter(raw_eeg, fs, 1.0, 40.0, 5)
        ########################################################################

        for i, mark_idx in enumerate(mark_indices):
            epoch[i, :] = clean_eeg[mark_idx + e_s : mark_idx + e_e] # grab the appropriate samples around the stimulus onset

        # Baseline correction
        if baseline:
            for i in range(0, int(epoch.shape[0])):
                epoch[i, :] = epoch[i, :] - np.mean(epoch[i, b_s:b_e])

        # stack epoch of each channel
        final_epoch = np.dstack((final_epoch, epoch))
    final_epoch = np.swapaxes(final_epoch, 1, 2)
    return final_epoch

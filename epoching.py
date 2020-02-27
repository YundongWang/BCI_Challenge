"""
@author: Yundong Wang, Zimu Li
@reference: UC San Diego COGS 189 Winter 2019 Assignment 1, writen by Alessandro D'Amico
"""
import numpy as np                                      # for dealing with data
from scipy.signal import butter, sosfiltfilt  # for filtering
import matplotlib.pyplot as plt                         # for plotting
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir


def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog = False, btype = 'band', output = 'sos')
    filted_data = sosfiltfilt(sos, data)
    return filted_data

def generate_epoch(label_col_name, file_path, channels, fs=200.0, lowcut=1.0, highcut=40.0, epoch_s = 0, epoch_e = 1300, bl_s = -400, bl_e = -300):
    """
    :label_col_name (String): column name for your target label in csv file.
    :file_path (String): path to your csv file
    :channels (int): number of channels from your EEG data
    :fs (float, optional): sampling rate
    :lowcut (float, optional): lowest frequency we will pass
    :highcut (float, optional): highest frequency we will pass
    :epoch_s (int, optional): epoch starting time relative to stmulus in miliseconds
    :epoch_e (int, optional): epoch ending time relative to stmulus in miliseconds
    :bl_s (int, optional): baseline starting time relative to stmulus in miliseconds
    :bl_e (int, optional): baseline ending time relative to stmulus in miliseconds
    :rtype (3d-nparray): epoched data with dimension (stimulus_per_participants, number_of_channels, number_of_time_points)
    """
    # read dataand data selection
    train_data = pd.read_csv(file_path)

    train_data.loc[:,'Time'] = train_data.loc[:,'Time']*1000
    raw_eeg_df  = train_data[channels].values.T

    time_df = train_data['Time'].values
    train_data['index'] = train_data.index.values
    mark_index_df = train_data[train_data[label_col_name] == 1][[label_col_name,'index']].values
    mark_df = train_data[train_data[label_col_name] == 1][[label_col_name,'Time']].values
    #data cleaning
    mark_df = mark_df.astype(int)         # convert from float -> int
    time_df = time_df.astype(int)         # convert from float -> int
    rounder = int(round(1 / (fs / 1000))) # determine the integer to round by
    for i in range(0, (mark_df.shape[0])): # we will iterate through the rows (shape[0]) of mark_df
        mark_df[i, 1] = (round(mark_df[i, 1] / rounder) * rounder) # we will round the time values

    order = 5

    # Define the bounds of our epoch as well as our baseline
    b_s = int((abs(epoch_s) + bl_s) * (fs / 1000)) # index in epoch_df where our baseline begins
    b_e = int((abs(epoch_s) + bl_e) * (fs / 1000)) # index in epoch_df where our baseline ends
    # Let's calculate the length our epoch with our given sampling rate
    epoch_len = int((abs(epoch_s) + abs(epoch_e)) * (fs / 1000))

    # Let's define some helpful variables to make our extraction easier
    e_s = int((epoch_s * (fs / 1000))) # effectively the number of indices before marker we want
    e_e = int((epoch_e * (fs / 1000))) # effectively the number of indices after marker we want
    # Epoch the data
    final_epoch = np.empty((mark_index_df.shape[0], epoch_len, 0), float)
    for channel in channels:
        epoch = np.zeros(shape = (int(mark_index_df.shape[0]), epoch_len))
        raw_eeg_df = train_data[channel].values
        clean_eeg_df = butter_bandpass_filter(raw_eeg_df, lowcut, highcut, fs, order)
        for i in range(0, int(mark_index_df.shape[0])):
            t = mark_index_df[i, 1] # the rounded time point of stimulus onset
            epoch[i, :] = clean_eeg_df[t + e_s:t + e_e] # grab the appropriate samples around the stimulus onset

        # Baseline correction
        for i in range(0, int(epoch.shape[0])):
            epoch[i, :] = epoch[i, :] - np.mean(epoch[i, b_s:b_e])
        final_epoch = np.dstack((final_epoch, epoch))
    final_epoch = np.swapaxes(final_epoch, 1, 2)
    return final_epoch

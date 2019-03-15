"""
@author: Yundong Wang, Zimu Li
"""
import numpy as np                                      # for dealing with data
from scipy.signal import butter, sosfiltfilt, sosfreqz  # for filtering
import matplotlib.pyplot as plt                         # for plotting
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace


# Create filtering variables
fs = 200.0     # 200 Hz sampling rate
lowcut = 1.0   # 1 Hz is the lowest frequency we will pass
highcut = 40.0 # 40  Hz is the highest frequency we will pass.

train_list_arr = np.array(sorted(listdir('./data/train')))
train_list_np = np.reshape(train_list_arr, (16,5))
test_list_arr = np.array(sorted(listdir('./data/test')))
test_list_np = np.reshape(test_list_arr, (10,5))

def butter_bandpass(lowcut, highcut, fs, order = 2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog = False, btype = 'band', output = 'sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
        sos = butter_bandpass(lowcut, highcut, fs, order = order)
        y = sosfiltfilt(sos, data)
        return y

def epoching(file_path):
    # read dataand data selection
    train_data = pd.read_csv(file_path)
    channels = np.load('channels.npy')

    train_data.loc[:,'Time'] = train_data.loc[:,'Time']*1000
    raw_eeg_df  = train_data[channels].values.T

    time_df = train_data['Time'].values
    train_data['index'] = train_data.index.values
    mark_index_df = train_data[train_data['FeedBackEvent'] == 1][['FeedBackEvent','index']].values
    mark_df = train_data[train_data['FeedBackEvent'] == 1][['FeedBackEvent','Time']].values
    #data cleaning
    mark_df = mark_df.astype(int)         # convert from float -> int
    time_df = time_df.astype(int)         # convert from float -> int
    rounder = int(round(1 / (fs / 1000))) # determine the integer to round by
    for i in range(0, (mark_df.shape[0])): # we will iterate through the rows (shape[0]) of mark_df
        mark_df[i, 1] = (round(mark_df[i, 1] / rounder) * rounder) # we will round the time values

    order = 5

    # Define the bounds of our epoch as well as our baseline
    epoch_s = 0
    epoch_e = 1300
    bl_s    = -400
    bl_e    = -300

    # Let's calculate the length our epoch with our given sampling rate
    epoch_len = int((abs(epoch_s) + abs(epoch_e)) * (fs / 1000))

    # Let's define some helpful variables to make our extraction easier
    e_s = int((epoch_s * (fs / 1000))) # effectively the number of indices before marker we want
    e_e = int((epoch_e * (fs / 1000))) # effectively the number of indices after marker we want
    # Epoch the data
    final_epoch = np.empty((mark_index_df.shape[0], 260, 0), float)
    for channel in channels:
        epoch = np.zeros(shape = (int(mark_index_df.shape[0]), epoch_len))
        raw_eeg_df = train_data[channel].values
        clean_eeg_df = butter_bandpass_filter(raw_eeg_df, lowcut, highcut, fs, order)
        for i in range(0, int(mark_index_df.shape[0])):
            t = mark_index_df[i, 1] # the rounded time point of stimulus onset
            epoch[i, :] = clean_eeg_df[t + e_s:t + e_e] # grab the appropriate samples around the stimulus onset
        b_s = int((abs(epoch_s) + bl_s) * (fs / 1000)) # index in epoch_df where our baseline begins
        b_e = int((abs(epoch_s) + bl_e) * (fs / 1000)) # index in epoch_df where our baseline ends

        # Baseline correction
        for i in range(0, int(epoch.shape[0])):
            epoch[i, :] = epoch[i, :] - np.mean(epoch[i, b_s:b_e])
        final_epoch = np.dstack((final_epoch, epoch))
    final_epoch = np.swapaxes(final_epoch, 1, 2)
    return final_epoch

##################### epoching all training and testing data ###################
train_data_list, test_data_list = np.empty((0, 340, 56, 260), float), np.empty((0, 340, 56, 260), float)
for i in range(16):
    subject_dir_list = train_list_np[i]
    subject_epoch = np.empty((0, 56, 260), float)
    for j in range(5):
        subject_dir = subject_dir_list[j]
        data = epoching('./data/train/'+subject_dir)
        subject_epoch = np.vstack((subject_epoch, data))
    subject_epoch = np.reshape(subject_epoch, (1, 340, 56, 260))
    train_data_list = np.vstack((train_data_list, subject_epoch))

for i in range(10):
    subject_dir_list = test_list_np[i]
    subject_epoch = np.empty((0, 56, 260), float)
    for j in range(5):
        subject_dir = subject_dir_list[j]
        data = epoching('./data/test/'+subject_dir)
        subject_epoch = np.vstack((subject_epoch, data))
    subject_epoch = np.reshape(subject_epoch, (1, 340, 56, 260))
    test_data_list = np.vstack((test_data_list, subject_epoch))

print('Epoched training data shape: '+ str(train_data_list.shape))
print('Epoched testing data shape: '+ str(test_data_list.shape))

########################## apply data preprocessing ############################
y_train = pd.read_csv('TrainLabels.csv')['Prediction'].values
y_test = np.reshape(pd.read_csv('true_labels.csv', header=None).values, 3400)
XC = XdawnCovariances(nfilter=5)
output_train = XC.fit_transform(np.reshape(train_data_list, (16*340, 56, 260)), y_train)
X_train = TangentSpace(metric='riemann').fit_transform(output_train)
output_test = XC.fit_transform(np.reshape(test_data_list, (10*340, 56, 260)), y_test)
X_test = TangentSpace(metric='riemann').fit_transform(output_test)
print('Preprocessed training data shape: '+ str(X_train.shape))
print('Preprocessed testing data shape: '+ str(X_test.shape))

############################## save data to disk ###############################
np.save('./data/train_data_56_260_1_40Hz.npy', train_data_list)
np.save('./data/test_data_56_260_1_40Hz.npy', test_data_list)
np.save('./data/X_train', X_train)
np.save('./data/X_test', X_test)

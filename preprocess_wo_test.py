"""
@author: Yundong Wang, Zimu Li

Preprocessing the data from Kaggle BCI Challenge.
"""
from generate_epoch import *


def butter_bandpass_filter(raw_data, lowcut, highcut, fs, order = 2):
    '''
    The filter I want to apply to my raw eeg data.
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog = False, btype = 'band', output = 'sos')
    filted_data = sosfiltfilt(sos, raw_data)
    return filted_data

if __name__ == "__main__":
    total_training_subj = 16
    total_testing_subj = 10
    stimulus_per_subj = 340
    trial_per_subj = 5
    # Create filtering variables
    fs = 200.0     # 200 Hz sampling rate
    lowcut = 1.0   # 1 Hz is the lowest frequency we will pass
    highcut = 40.0 # 40  Hz is the highest frequency we will pass.
    epoch_s = 0
    epoch_e = 1300
    bl_s = -400
    bl_e = -300
    epoch_len = int((abs(epoch_s) + abs(epoch_e)) * (fs / 1000))
    channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
           'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
           'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
           'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
           'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
           'PO7', 'POz', 'P08', 'O1', 'O2']

    train_list_arr = np.array(sorted(listdir('./data/train')))
    train_list_np = np.reshape(train_list_arr, (total_training_subj,trial_per_subj))

    ##################### epoching all training data ###################
    train_data_list = np.empty((0, stimulus_per_subj, len(channels), epoch_len), float)

    # Iterate through all subjects
    for training_subj_id in range(total_training_subj):
        subject_dir_list = train_list_np[training_subj_id]
        subject_epoch = np.empty((0, len(channels), epoch_len), float)

        # iterate through all trials and generate epoch
        for trial_id in range(trial_per_subj):
            subject_dir = subject_dir_list[trial_id]
            data = generate_epoch('FeedBackEvent', './data/train/'+subject_dir, \
                channels, butter_bandpass_filter, True, fs, lowcut, highcut, epoch_s, epoch_e, bl_s, bl_e)
            subject_epoch = np.vstack((subject_epoch, data))
        subject_epoch = np.reshape(subject_epoch, (1, stimulus_per_subj, len(channels), epoch_len))
        train_data_list = np.vstack((train_data_list, subject_epoch))

    print('Epoched training data shape: '+ str(train_data_list.shape))

    # ############################## save data to disk ###############################
    np.save('./data/train_data.npy', train_data_list)

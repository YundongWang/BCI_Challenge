"""
@author: Yundong Wang, Zimu Li
"""
from epoching import *
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

if __name__ == "__main__":
    total_training_participant = 16
    total_testing_participant = 10
    stimulus_per_participants = 340
    trial_per_participants = 5
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
    train_list_np = np.reshape(train_list_arr, (total_training_participant,5))
    test_list_arr = np.array(sorted(listdir('./data/test')))
    test_list_np = np.reshape(test_list_arr, (total_testing_participant,5))

    ##################### epoching all training and testing data ###################
    train_data_list = np.empty((0, stimulus_per_participants, len(channels), epoch_len), float)
    test_data_list = np.empty((0, stimulus_per_participants, len(channels), epoch_len), float)

    for training_participant_id in range(total_training_participant):
        subject_dir_list = train_list_np[training_participant_id]
        subject_epoch = np.empty((0, len(channels), epoch_len), float)
        for trial_id in range(trial_per_participants):
            subject_dir = subject_dir_list[trial_id]
            data = generate_epoch('FeedBackEvent', './data/train/'+subject_dir, channels)
            subject_epoch = np.vstack((subject_epoch, data))
        subject_epoch = np.reshape(subject_epoch, (1, stimulus_per_participants, len(channels), epoch_len))
        train_data_list = np.vstack((train_data_list, subject_epoch))

    print('Epoched training data shape: '+ str(train_data_list.shape))

    for testing_participant_id in range(total_testing_participant):
        subject_dir_list = test_list_np[testing_participant_id]
        subject_epoch = np.empty((0, len(channels), epoch_len), float)
        for trial_id in range(trial_per_participants):
            subject_dir = subject_dir_list[trial_id]
            data = generate_epoch('FeedBackEvent', './data/test/'+subject_dir, channels)
            subject_epoch = np.vstack((subject_epoch, data))
        subject_epoch = np.reshape(subject_epoch, (1, stimulus_per_participants, len(channels), epoch_len))
        test_data_list = np.vstack((test_data_list, subject_epoch))

    print('Epoched testing data shape: '+ str(test_data_list.shape))

    # ########################## apply data preprocessing ############################
    y_train = pd.read_csv('data/TrainLabels.csv')['Prediction'].values
    XC = XdawnCovariances(nfilter=5)
    X_train = XC.fit_transform(np.reshape(train_data_list, (16*stimulus_per_participants, len(channels), epoch_len)), y_train)
    X_train = TangentSpace(metric='riemann').fit_transform(X_train)
    X_test = XC.transform(np.reshape(test_data_list, (10*stimulus_per_participants, len(channels), epoch_len)))
    X_test = TangentSpace(metric='riemann').transform(X_test)
    print('Preprocessed training data shape: '+ str(X_train.shape))
    print('Preprocessed testing data shape: '+ str(X_test.shape))

    # ############################## save data to disk ###############################
    np.save('./data/train_data.npy', train_data_list)
    np.save('./data/test_data.npy', test_data_list)
    np.save('./data/X_train', X_train)
    np.save('./data/X_test', X_test)

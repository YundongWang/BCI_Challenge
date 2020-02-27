from generate_epoch import *
import matplotlib.pyplot as plt               # for plotting


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

    channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
           'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
           'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
           'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
           'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
           'PO7', 'POz', 'P08', 'O1', 'O2']

    data = generate_epoch('FeedBackEvent', 'Data_S02_Sess01.csv', channels, butter_bandpass_filter)
    print('Epoched data shape: '+ str(data.shape)) # should be (60, 56, 180): 60 events, 56 channels, 180 time-stamps
    epoch_s = -100
    epoch_e = 800
    fs = 200.0
    dt = int(1000/fs)
    times = range(epoch_s, epoch_e, dt)

    # We want to draw Pz
    channel = channels.index("Cz")
    for i in range(int(data.shape[0])):
        plt.plot(times, data[i, channel, :])

    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.grid(True)
    plt.savefig('%s.png'%"Cz")

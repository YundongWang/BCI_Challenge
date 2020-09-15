"""
@author: Yundong Wang, Zimu Li
"""
from __future__ import division  # for python 2.7 only

import numpy as np
import pandas as pd
from EEGModels import EEGNet
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
X_train_valid = np.load('./data/train_data_56_260_1_40Hz.npy')
X_train_valid = np.reshape(X_train_valid, (16*340, 56, 260))

y_train_valid = pd.read_csv('./data/TrainLabels.csv')['Prediction'].values

X_test = np.load('./data/test_data_56_260_1_40Hz.npy')
X_test = np.reshape(X_test, (3400, 56, 260))

y_test = np.reshape(pd.read_csv(
    './data/true_labels.csv', header=None).values, 3400)

# data partition
X_train = X_train_valid[1360:, :]
X_valid = X_train_valid[:1360, :]
y_train = y_train_valid[1360:]
y_valid = y_train_valid[:1360]

kernels, chans, samples = 1, 56, 260

X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
X_valid = X_valid.reshape(X_valid.shape[0], kernels, chans, samples)
X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)

print(str(X_train.shape[0]) + ' train samples')
print(str(X_valid.shape[0]) + ' validation samples')
print(str(X_test.shape[0]) + ' test samples')

# configure EEGNET model
model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16,
               dropoutType='Dropout')

# compile the model and set the optimizers
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)

# Weighted loss
weight_0 = 1/(len([y for y in y_train_valid if y == 0]))
weight_1 = 1/(len([y for y in y_train_valid if y == 1]))
class_weights = {0: weight_0, 1: weight_1}
#
# # fit the model
fittedModel = model.fit(X_train, y_train, batch_size=34, epochs=100,
                        verbose=2, validation_data=(X_valid, y_valid),
                        callbacks=[checkpointer], class_weight=class_weights)

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

# Evaluate
y_probs = model.predict(X_test)
y_pred = y_probs.argmax(axis=-1)

# save score
csv = pd.read_csv('./data/benchmark.csv')
csv['Prediction'] = y_probs
csv.to_csv('./submission/submissionEEGNET.csv', index=False)
print('--------------------Submission file has been generated.--------------------------')

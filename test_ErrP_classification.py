import numpy as np
import random

import mne
from mne import io

# EEGNet-specific imports
from myFrame import my_EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import itertools
import os

"""""
    The dataset used in this project is a pibilc Monitoring Error-related potentials detaset.
    You can find it with the dataset: 
        http://bnci-horizon-2020.eu/database/data-sets.
    The detailed description of the dataset is shown in 
        https://lampx.tugraz.at/~bci/database/013-2015/description.pdf.
"""""

##################################load sub1 data ###############################
data_path = 'C:/Users/Pursue_Lee/Desktop/error-related potentials ' \
            'dataset/error related potentials/subjects'
sub1_raw_fname = 'sub1_afterICA_raw.fif'
sub1_event_fname = data_path+ '/Subject 1/Merged Run/sub1_1_eve.txt'
sub1_events = mne.read_events(sub1_event_fname)
#print(events)

#Merge event into two classes: error & correct
sub1_merged_events=mne.pick_events(sub1_events, include=[5,6,9,10])
sub1_merged_events=mne.merge_events(sub1_merged_events,[5,10],1)
sub1_merged_events=mne.merge_events(sub1_merged_events,[6,9],2)
events_dict={'correct movement':1,'erronemous movement':2}

"""""
    #Plot afterICA raw data with events:11 marks correct movement; 12 marks erroneous movement
"""""
raw=io.read_raw_fif(sub1_raw_fname, preload=True, verbose=False)
raw.plot(events=sub1_merged_events,n_channels=10,start=20,duration=80,
         color='b',event_color={1:'g',2:'r'})

#Extract the epochs. tmin and tmax mean the starting time and ending time of each epoch.
tmin, tmax=-0.1, 0.5
sub1_epochs_notResample = mne.Epochs(raw, sub1_merged_events,tmin=-0.1, tmax=0.5,event_id=events_dict,
                    proj=False,baseline=None, preload=True, verbose=False)
sub1_epochs = sub1_epochs_notResample.copy().resample(128, npad='auto')
sub1_labels = sub1_epochs.events[:, -1]
print(sub1_epochs)
#print(labels)

# extract raw data. scale by 10000 due to scaling sensitivity in deep learning
sub1_X = sub1_epochs.get_data() * 10000   # format is in (trials, channels, samples)
print(sub1_X.shape)
#print(X[1,:,:])
sub1_y = sub1_labels

############################## load sub2 data###########################
sub2_raw_fname = 'sub2_afterICA_raw.fif'
sub2_event_fname = data_path+ '/Subject 2/sub2_1_eve.txt'
sub2_events = mne.read_events(sub2_event_fname)
#print(events)

#Merge event into two classes: error & correct
sub2_merged_events=mne.pick_events(sub2_events, include=[5,6,9,10])
sub2_merged_events=mne.merge_events(sub2_merged_events,[5,10],1)
sub2_merged_events=mne.merge_events(sub2_merged_events,[6,9],2)
events_dict={'correct movement':1,'erronemous movement':2}

"""""
    #Plot afterICA raw data with events:11 marks correct movement; 12 marks erroneous movement
"""""
raw=io.read_raw_fif(sub2_raw_fname, preload=True, verbose=False)
raw.plot(events=sub2_merged_events,n_channels=10,start=20,duration=80,
         color='b',event_color={1:'g',2:'r'})

#Extract the epochs. tmin and tmax mean the starting time and ending time of each epoch.
tmin, tmax=-0.1, 0.5
sub2_epochs_notResample = mne.Epochs(raw, sub2_merged_events,tmin=-0.1, tmax=0.5,event_id=events_dict,
                    proj=False,baseline=None, preload=True, verbose=False)
sub2_epochs = sub2_epochs_notResample.copy().resample(128, npad='auto')
sub2_labels = sub2_epochs.events[:, -1]
print(sub2_epochs)
#print(labels)

# extract raw data. scale by 10000 due to scaling sensitivity in deep learning
sub2_X = sub2_epochs.get_data() * 10000   # format is in (trials, channels, samples)
print(sub2_X.shape)
#print(X[1,:,:])
sub2_y = sub2_labels

################################ load sub3 data ##############################
sub3_raw_fname = 'sub3_afterICA_raw.fif'
sub3_event_fname = data_path+ '/Subject 3/sub3_1_eve.txt'
sub3_events = mne.read_events(sub3_event_fname)
#print(events)

#Merge event into two classes: error & correct
sub3_merged_events=mne.pick_events(sub3_events, include=[5,6,9,10])
sub3_merged_events=mne.merge_events(sub3_merged_events,[5,10],1)
sub3_merged_events=mne.merge_events(sub3_merged_events,[6,9],2)
events_dict={'correct movement':1,'erronemous movement':2}

"""""
    #Plot afterICA raw data with events:11 marks correct movement; 12 marks erroneous movement
"""""
raw=io.read_raw_fif(sub3_raw_fname, preload=True, verbose=False)
raw.plot(events=sub3_merged_events,n_channels=10,start=20,duration=80,
         color='b',event_color={1:'g',2:'r'})

#Extract the epochs. tmin and tmax mean the starting time and ending time of each epoch.
tmin, tmax=-0.1, 0.5
sub3_epochs_notResample = mne.Epochs(raw, sub3_merged_events,tmin=-0.1, tmax=0.5,event_id=events_dict,
                    proj=False,baseline=None, preload=True, verbose=False)
sub3_epochs = sub3_epochs_notResample.copy().resample(128, npad='auto')
sub3_labels = sub3_epochs.events[:, -1]
print(sub3_epochs)
#print(labels)

# extract raw data. scale by 10000 due to scaling sensitivity in deep learning
sub3_X = sub3_epochs.get_data() * 10000   # format is in (trials, channels, samples)
print(sub3_X.shape)
#print(X[1,:,:])
sub3_y = sub3_labels

################################ load sub5 data ##############################
sub5_raw_fname = 'sub5_afterICA_raw.fif'
sub5_event_fname = data_path+ '/Subject 5/sub5_1_eve.txt'
sub5_events = mne.read_events(sub5_event_fname)
#print(events)

#Merge event into two classes: error & correct
sub5_merged_events=mne.pick_events(sub5_events, include=[5,6,9,10])
sub5_merged_events=mne.merge_events(sub5_merged_events,[5,10],1)
sub5_merged_events=mne.merge_events(sub5_merged_events,[6,9],2)
events_dict={'correct movement':1,'erronemous movement':2}

"""""
    #Plot afterICA raw data with events:11 marks correct movement; 12 marks erroneous movement
"""""
raw=io.read_raw_fif(sub5_raw_fname, preload=True, verbose=False)
raw.plot(events=sub5_merged_events,n_channels=10,start=20,duration=80,
         color='b',event_color={1:'g',2:'r'})

#Extract the epochs. tmin and tmax mean the starting time and ending time of each epoch.
tmin, tmax=-0.1, 0.5
sub5_epochs_notResample = mne.Epochs(raw, sub5_merged_events,tmin=-0.1, tmax=0.5,event_id=events_dict,
                    proj=False,baseline=None, preload=True, verbose=False)
sub5_epochs = sub5_epochs_notResample.copy().resample(128, npad='auto')
sub5_labels = sub5_epochs.events[:, -1]
print(sub5_epochs)
#print(labels)

# extract raw data. scale by 10000 due to scaling sensitivity in deep learning
sub5_X = sub5_epochs.get_data() * 10000   # format is in (trials, channels, samples)
print(sub5_X.shape)
#print(X[1,:,:])
sub5_y = sub5_labels

################################ load sub6 data ##############################
sub6_raw_fname = 'sub6_afterICA_raw.fif'
sub6_event_fname = data_path+ '/Subject 6/sub6_1_eve.txt'
sub6_events = mne.read_events(sub6_event_fname)
#print(events)

#Merge event into two classes: error & correct
sub6_merged_events=mne.pick_events(sub6_events, include=[5,6,9,10])
sub6_merged_events=mne.merge_events(sub6_merged_events,[5,10],1)
sub6_merged_events=mne.merge_events(sub6_merged_events,[6,9],2)
events_dict={'correct movement':1,'erronemous movement':2}

"""""
    #Plot afterICA raw data with events:11 marks correct movement; 12 marks erroneous movement
"""""
raw=io.read_raw_fif(sub6_raw_fname, preload=True, verbose=False)
raw.plot(events=sub6_merged_events,n_channels=10,start=20,duration=80,
         color='b',event_color={1:'g',2:'r'})

#Extract the epochs. tmin and tmax mean the starting time and ending time of each epoch.
tmin, tmax=-0.1, 0.5
sub6_epochs_notResample = mne.Epochs(raw, sub6_merged_events,tmin=-0.1, tmax=0.5,event_id=events_dict,
                    proj=False,baseline=None, preload=True, verbose=False)
sub6_epochs = sub6_epochs_notResample.copy().resample(128, npad='auto')
sub6_labels = sub6_epochs.events[:, -1]
print(sub6_epochs)
#print(labels)

# extract raw data. scale by 10000 due to scaling sensitivity in deep learning
sub6_X = sub6_epochs.get_data() * 10000   # format is in (trials, channels, samples)
print(sub6_X.shape)
#print(X[1,:,:])
sub6_y = sub6_labels


#print(y)
#print(y.shape)

X = np.concatenate((sub1_X, sub2_X, sub3_X, sub5_X, sub6_X), axis=0)
y = np.concatenate((sub1_y, sub2_y, sub3_y, sub5_y, sub6_y), axis=0)
print(X.shape)
print(y.shape)

#shuffle the data
index = [i for i in range(X.shape[0])]
random.shuffle(index)
X = X[index]
y = y[index]



"""""
#replicate the data
X=np.tile(X,(5,1,1))
#print(X.shape)
y=np.tile(y,5)
#print(y.shape)
#print(y)
"""""
kernels,chans,samples=1,64,77

#np.random.shuffle(X)
#print(X.shape)
#print(X[1,:,:])

# take 60/20/20 percent of the data to train/validate/test
X_train = X[0:1568, ]
#print(X_train.shape)
Y_train = y[0:1568]
X_validate = X[1568:2091, ]
Y_validate = y[1568:2091]
X_test = X[2091:, ]
Y_test = y[2091:]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train - 1)
Y_validate = np_utils.to_categorical(Y_validate - 1)
Y_test = np_utils.to_categorical(Y_test - 1)

np.savetxt('y_test.txt',Y_test.argmax(axis=-1))

# convert data to NCHW (trials, kernels, channels, samples) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
model = my_EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=32, F1=8, D=1, F2=16,
               dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


# count number of parameters in the model
numParams = model.count_params()


model_path = 'mytest_2.h5'

model.save(model_path)

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='mytest_2.h5', monitor='val_accuracy',
                               verbose=1, mode='max',
                               save_best_only=True)

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# class1: 1, class2: 3
class_weights = {0:1,1:3.65}

################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size=32, epochs=600, shuffle=True,
                        verbose=2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight=class_weights)

# load optimal weights
model.load_weights('C:\\Users\\Pursue_Lee\\PycharmProjects\\Error realted potentials\\mytest_2.h5')

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
np.savetxt('y_true.txt',preds)
print("Classification accuracy: %f " % (acc))
print(numParams)


# 绘制训练 & 验证的准确率值
plt.plot(fittedModel.history['accuracy'])
plt.plot(fittedModel.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(fittedModel.history['loss'])
plt.plot(fittedModel.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
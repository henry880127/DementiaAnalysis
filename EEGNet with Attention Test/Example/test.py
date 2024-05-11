# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:44:03 2023

@author: NESS4090
"""

import mat73
import scipy
import scipy.io as sio # cannot use for v7.3 mat file
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import pickle
from EEGNet_function import EEGNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import pandas as pd

#%% filtering
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, lowcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  b, a = butter(order, low, btype='low')
  y = filtfilt(b, a, data) # zero-phase filter # data: [ch x time]
  return y

def butter_highpass_filter(data, highcut, fs, order):
  nyq = fs/2
  high = highcut/nyq
  b, a = butter(order, high, btype='high')
  y = filtfilt(b, a, data) # zero-phase filter
  return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order, [low, high], btype='band')
  # demean before filtering
  meandat = np.mean(data, axis=1)
  data = data - meandat[:, np.newaxis]
  y = filtfilt(b, a, data)
  return y

#%% re-sample
def down_sample(data,ogFs,targetFreq=256):
    q = int(np.floor(ogFs/targetFreq))
    data = signal.decimate(data,q,axis=1)
    return data

#%%
def extractEpoch3D(data, event, baseline, frame, opt_keep_baseline, ogFs, targetFreq=250, bandPassFreqs=[0.5,100]):
  # extract epoch from 2D data into 3D [ch x time x trial]
  # input: event, baseline, frame
  # extract epoch = baseline[0] to frame[2]
  q = int(np.floor(ogFs/targetFreq))
  event = (event/q).astype(int)
  data = down_sample(data, ogFs, targetFreq=targetFreq)
  data = butter_bandpass_filter(data, bandPassFreqs[0], bandPassFreqs[1], targetFreq, 4)
  # for memory pre-allocation
  if opt_keep_baseline == True:
    begin_tmp = int(np.floor(baseline[0]/1000*targetFreq))
    end_tmp = int(begin_tmp+np.floor(frame[1]-baseline[0])/1000*targetFreq)
  else:
    begin_tmp = int(np.floor(frame[0]/1000*targetFreq))
    end_tmp = int(begin_tmp+np.floor(frame[1]-frame[0])/1000*targetFreq)
  
  epoch3D = np.zeros((data.shape[0], end_tmp-begin_tmp, len(event)))
  nth_event = 0

  for i in event:
    if opt_keep_baseline == True:
      begin_id = int(i + np.floor(baseline[0]/1000 * targetFreq))
      end_id = int(begin_id + np.floor((frame[1]-baseline[0])/1000*targetFreq))
    else:
      begin_id = int(i + np.floor(frame[0]/1000 * targetFreq))
      end_id = int(begin_id + np.floor((frame[1]-frame[0])/1000*targetFreq))
    
    tmp_data = data[:, begin_id:end_id]

    begin_base = int(np.floor(baseline[0]/1000 * targetFreq))
    end_base = int(begin_base + np.floor(np.diff(baseline)/1000 * targetFreq)-1)
    base = np.mean(tmp_data[:, begin_base:end_base], axis=1)

    rmbase_data = tmp_data - base[:, np.newaxis]
    epoch3D[:, :, nth_event] = rmbase_data
    nth_event = nth_event + 1

  return epoch3D
# event = np.array([1,2,3,4,5,6])
# print(event)
# event_div = event/3
# print(event_div)
# event_div = event_div.astype(int)
# print(event_div)

#%%
def loadPickle(pklDir):
    with open(pklDir,'rb') as fp:
        dd = pickle.load(fp)
        return dd
    
def reshape2Input(a):
    newshape = (a.shape[2], 1, a.shape[0], a.shape[1])  # trials , 1 , EEG_channels , sample_points
    return a.reshape(newshape)

def saveResult(df, save_dir):
    df.to_csv(save_dir)
    
def EEGNet_fit(folder_dir, pkl_name='unname', logs_dir='logs', K=None, startPt=0, epoch_length=1000):
    '''data'''
    dictData = loadPickle(f"{folder_dir}/{pkl_name}")
    targetEEG_train = reshape2Input(dictData['targetEEG_train'])[
        :, :, :, startPt:startPt+epoch_length]
    targetEEG_test = reshape2Input(dictData['targetEEG_test'])[
        :, :, :, startPt:startPt+epoch_length]
    nontargetEEG_train = reshape2Input(dictData['nontargetEEG_train'])[
        :, :, :, startPt:startPt+epoch_length]
    nontargetEEG_test = reshape2Input(dictData['nontargetEEG_test'])[
        :, :, :, startPt:startPt+epoch_length]
    epochs_train = np.concatenate((targetEEG_train, nontargetEEG_train), axis=0)
    epochs_test = np.concatenate((targetEEG_test, nontargetEEG_test), axis=0)
    
    '''Label'''
    encoder = OneHotEncoder(sparse=False)
    y_train = np.ones((targetEEG_train.shape[0]))
    y_train = np.concatenate((y_train, np.ones(nontargetEEG_train.shape[0])+1))
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = np.ones((targetEEG_test.shape[0]))
    y_test = np.concatenate((y_test, np.ones(nontargetEEG_test.shape[0])+1))
    y_test = encoder.fit_transform(y_test.reshape(-1, 1))
    
    '''shuffle'''
    num_samples = epochs_train.shape[0]
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)
    epochs_train = epochs_train[shuffled_indices, :, :, :]
    y_train = y_train[shuffled_indices,:]
    num_samples = epochs_test.shape[0]
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)
    epochs_test = epochs_test[shuffled_indices, :, :, :]
    y_test = y_test[shuffled_indices,:]
    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)
    print('epochs_train.shape:', epochs_train.shape)
    print('epochs_test.shape:', epochs_test.shape)
    savingFoldername = 'data_check'
    saveResult(pd.DataFrame(y_train), f'./results/{savingFoldername}/y_train.csv')
    saveResult(pd.DataFrame(y_test), f'./results/{savingFoldername}/y_test.csv')
    # saveResult(pd.DataFrame(epochs_train), f'./results/{savingFoldername}/epochs_train.csv')
    # saveResult(pd.DataFrame(epochs_test), f'./results/{savingFoldername}/epochs_test.csv')
    model = EEGNet(input_shape=(1, epochs_train.shape[2], epochs_train.shape[3]),
                   loss='categorical_crossentropy',
                   epochs=100,
                   batch_size=300,
                   kernLength=125,  # Half of sampling rate
                   lr=0.0005,
                   min_lr=0.0001,
                   log_path=f'{logs_dir}/log_{pkl_name}',
                   model_name=f'EEGNet_{pkl_name}',
                   F1=16,
                   avgPoolSize_b1=(1, 8),
                   avgPoolSize_b2=(1, 8))
    history_list = list()
    if (K == None):
        # history = model.fit(epochs_train, y_train, epochs_test, y_test)
        history = model.fit(epochs_test, y_test, epochs_train, y_train)
        history_list.append(history)
        return history_list
    else:
        dataSet = np.concatenate([epochs_train, epochs_test])
        y = np.concatenate([y_train, y_test])
        print(f'KFold Applied! dataSet:{dataSet.shape}  y:{y.shape}')
        kf = KFold(n_splits=K)
        kf.get_n_splits(dataSet, y)
        print(kf)
        for i, (train_index, test_index) in enumerate(kf.split(dataSet)):
            epochs_train = dataSet[train_index, :, :, :]
            epochs_test = dataSet[test_index, :, :, :]
            y_train = y[train_index, :]
            y_test = y[test_index, :]
            history = model.fit(epochs_train, y_train, epochs_test, y_test)
            history_list.append(history)
        return history_list

def EEGNet_fit_specificCHs(folder_dir, pkl_name='unname', logs_dir='logs', K=None, startPt=0, epoch_length=1000):
    '''data'''
    ind_CHs = [31]
    dictData = loadPickle(f"{folder_dir}/{pkl_name}")
    targetEEG_train = reshape2Input(dictData['targetEEG_train'])[
        :, :, ind_CHs, startPt:startPt+epoch_length]
    targetEEG_test = reshape2Input(dictData['targetEEG_test'])[
        :, :, ind_CHs, startPt:startPt+epoch_length]
    nontargetEEG_train = reshape2Input(dictData['nontargetEEG_train'])[
        :, :, ind_CHs, startPt:startPt+epoch_length]
    nontargetEEG_test = reshape2Input(dictData['nontargetEEG_test'])[
        :, :, ind_CHs, startPt:startPt+epoch_length]
    epochs_train = np.concatenate(
        (targetEEG_train, nontargetEEG_train), axis=0)
    epochs_test = np.concatenate((targetEEG_test, nontargetEEG_test), axis=0)
    
    '''Label'''
    encoder = OneHotEncoder(sparse=False)
    y_train = np.ones((targetEEG_train.shape[0]))
    y_train = np.concatenate((y_train, np.ones(nontargetEEG_train.shape[0])+1))
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = np.ones((targetEEG_test.shape[0]))
    y_test = np.concatenate((y_test, np.ones(nontargetEEG_test.shape[0])+1))
    y_test = encoder.fit_transform(y_test.reshape(-1, 1))
    
    '''shuffle'''
    num_samples = epochs_train.shape[0]
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)
    epochs_train = epochs_train[shuffled_indices, :, :, :]
    y_train = y_train[shuffled_indices, :]
    num_samples = epochs_test.shape[0]
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)
    epochs_test = epochs_test[shuffled_indices, :, :, :]
    y_test = y_test[shuffled_indices, :]
    print('epochs_train.shape:', epochs_train.shape)
    print('epochs_test.shape:', epochs_test.shape)
    print('y_train:', y_train.shape)
    print('y_test.shape:', y_test.shape)
    shuffled_indices = np.arange(num_samples)
    model = EEGNet(input_shape=(1, epochs_train.shape[2], epochs_train.shape[3]),
                   loss='categorical_crossentropy',
                   epochs=40,
                   batch_size=300,
                   kernLength=125,  # Half of sampling rate
                   lr=0.0005,
                   min_lr=0.0001,
                   log_path=f'{logs_dir}/log_{pkl_name}',
                   model_name=f'EEGNet_{pkl_name}',
                   F1=16,
                   avgPoolSize_b1=(1, 8),
                   avgPoolSize_b2=(1, 8))
    history_list = list()
    if (K == None):
        history = model.fit(epochs_train, y_train, epochs_test, y_test)
        history_list.append(history)
        return history_list
    else:
        dataSet = np.concatenate([epochs_train, epochs_test])
        y = np.concatenate([y_train, y_test])
        print(f'KFold Applied! dataSet:{dataSet.shape}  y:{y.shape}')
        kf = KFold(n_splits=K)
        kf.get_n_splits(dataSet, y)
        print(kf)
        for i, (train_index, test_index) in enumerate(kf.split(dataSet)):
            epochs_train = dataSet[train_index, :, :, :]
            epochs_test = dataSet[test_index, :, :, :]
            y_train = y[train_index, :]
            y_test = y[test_index, :]
            history = model.fit(epochs_train, y_train, epochs_test, y_test)
            history_list.append(history)
        return history_list

def EEGNet_fit_folder(folder_dir, logs_dir='logs', ch_set='all', K=None, startPt=0,
                      epoch_length=1000, savingFoldername='folder', pltfilename='sess'):
    data_list = os.listdir(folder_dir)[0:3]
    history_list = list()
    f, ax = plt.subplots(8, 7, figsize=(30, 30))
    ax = ax.ravel()
    for i, pkl in enumerate(data_list):
        pickle_dir = f'{folder_dir}/{pkl}'
        if (ch_set == '20ch'):
            history = EEGNet_fit_specificCHs(folder_dir, pkl_name=pkl, logs_dir=logs_dir,
                                      startPt=startPt, epoch_length=epoch_length, K=K)
        else:
            history = EEGNet_fit(folder_dir, pkl_name=pkl, logs_dir=logs_dir,
                                 startPt=startPt, epoch_length=epoch_length, K=K)
        history_list.append(history)

        '''Loss Curve Plot'''
        for j, fold in enumerate(history):
            if (j == 0):
                ax[i].plot(fold.history['loss'], '-', color='#1f77b4',
                           label='train')  # loss presents the train_loss
                ax[i].plot(fold.history['val_loss'], '-',
                           color='#ff7f0e', label='validation')
            else:
                # loss presents the train_loss
                ax[i].plot(fold.history['loss'], '-', color='#1f77b4',)
                ax[i].plot(fold.history['val_loss'], '-', color='#ff7f0e',)
            pkl_no_ext = ".".join(pkl.split(".")[:-1])
            ax[i].set_title(pkl_no_ext)
            ax[i].set_ylabel('loss')
            # ax[i].set_xlim([0,40])
            ax[i].set_ylim([0, 1.5])
            ax[i].set_yticks([0, 0.5, 1.0, 1.5])
            ax[i].set_xlabel('#iter')
    ax[-1].legend(bbox_to_anchor=(-5., -0.4, 2, .102),
                  loc=0, ncol=3, mode="expand", borderaxespad=0)
    plt.tight_layout()
    if not os.path.isdir(f'./results/{savingFoldername}'):
        os.mkdir(f'./results/{savingFoldername}')
    plt.savefig(f'./results/{savingFoldername}/{pltfilename}.png')
    return history_list, ax


folder_dir = './organized'
data_list = os.listdir(folder_dir)
# pkl_name = data_list[0]
frame = [0, 150]
startPt = frame[0]
epoch_length = frame[1]-frame[0]
K = None
logs_dir = 'logs'
savingFoldername = 'precision_test'
filename = 'noCV_shuffle_100epochs_traintestswitch'

history_list, lossCurve = EEGNet_fit_folder(folder_dir, logs_dir='logs', ch_set='all', K=K, startPt=startPt,
                                            epoch_length=epoch_length, savingFoldername=savingFoldername, pltfilename=filename)


last_result = list()
for i, history in enumerate(history_list):
    avg_fold = pd.Series(
        [0, 0, 0, 0], ['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    for j, fold in enumerate(history):
        DF = pd.DataFrame(fold.history)
        avg_fold += DF.iloc[-1, :]
    avg_fold = avg_fold / (j+1)
    last_result.append(avg_fold)
last_result = pd.DataFrame(last_result, index=None)
last_result = last_result.reset_index(drop=True)


if not os.path.isdir(f'./results/{savingFoldername}'):
    os.mkdir(f'./results/{savingFoldername}')
saveResult(last_result, f'./results/{savingFoldername}/{filename}.csv')
plt.show()
del lossCurve
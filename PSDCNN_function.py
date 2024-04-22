# -*- coding: utf-8 -*-
"""
Edited by Kuan 2023.11.09
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
# import tensorflow_model_analysis as tfma

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score
from matplotlib import pyplot as plt
import datetime

# custom metrics:https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def balanced_accuracy(y_true, y_pred):
    recall_keras = recall(y_true, y_pred)
    specificity_val = specificity(y_true, y_pred)
    return (specificity_val+recall_keras)/(2.0 + + K.epsilon())


class EEGNet:
    def __init__(self,
                 input_shape=(1, 30, 90),
                 num_class=2,
                 loss='categorical_crossentropy',
                 epochs=200,
                 batch_size=100,
                 optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),  # For the case of using EEGNet nultiple times, create the Adam and input to the initializer.
                 lr=0.01,
                 min_lr=0.01,
                 factor=0.25,
                 patience=10,
                 es_patience=20,
                 verbose=1,
                 log_path='logs',
                 log_path_TB='logs_TB',
                 date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                 model_name='EEGNet',
                 **kwargs):
        self.input_shape = input_shape
        self.num_class = num_class
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer.lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.log_path_TB = log_path_TB
        self.model_name = model_name
        self.weights_dir = log_path+'/'+model_name+'_out_.weights.h5'
        self.csv_dir = log_path+'/'+model_name+'_out_log.log'
        self.time_log = log_path+'/'+model_name+'_time_log.csv'

        # use **kwargs to set the new value of below args.
        self.kernLength1 = 3
        self.kernLength2 = 3
        self.num_Dense = 300
        self.strides = (1, 1)
        self.K = 16
        self.D = 2
        self.F2 = int(self.F1*self.D)
        self.norm_rate = 0.25
        self.dropout_rate = 0.5
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.data_format = 'channels_first'
        self.shuffle = False
        # self.metrics = 'accuracy'
        self.metrics = ['accuracy']
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weights_only = True
        self.seed = 1234
        self.class_balancing = False
        self.class_weight = None
        self.maxPoolSize = (3, 3)
        self.maxPoolStride = (2, 1)

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        input1 = Input(shape=self.input_shape)

        ##################################################################
        block1 = Conv2D(self.K, (self.kernLength1, self.kernLength2), padding='same', strides=self.strides,
                        input_shape=self.input_shape, name='Conv2D',
                        use_bias=True)(input1)
        block1 = Activation('relu', name='Act')(block1)
        block1 = MaxPooling2D(self.maxPoolSize, strides=self.maxPoolStride, name='maxPool')(block1)
        flatten = Flatten(name='flatten')(block1)

        dense1 = Dense(self.num_Dense, name='dense1')(flatten)
        dense2 = Dense(self.num_class, name='dense2')(dense1)
        softmax = Activation('softmax', name='softmax')(dense2)

        return Model(inputs=input1, outputs=softmax)

    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 4:
            raise Exception(
                'ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception(
                'ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        self.input_shape = X_train.shape[1:]
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]

        csv_logger = CSVLogger(self.csv_dir)
        # time_callback = TimeHistory(self.time_log)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, verbose=self.verbose,
                                       save_best_only=self.save_best_only, save_weights_only=self.save_weights_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, factor=self.factor,
                                      mode=self.mode, verbose=self.verbose, min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode,
                           verbose=self.verbose, patience=self.es_patience)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_path_TB, histogram_freq=1)

        model = self.build()
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=self.metrics)
        model.summary()
        print("The first kernel size is (1, {})".format(self.kernLength))

        # if self.class_balancing: # compute_class_weight if class_balancing is True
        #     self.class_weight = compute_class_weight(y_train)
        # else:
        #     self.class_weight = None
        # history = model.fit(X_train, y_train,
        #                     batch_size=self.batch_size,
        #                     shuffle=self.shuffle, epochs=self.epochs,
        #                     validation_data=(X_val, y_val))  # debug:刪除callbacks中的'es'
        history = model.fit(X_train, y_train,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle, epochs=self.epochs,
                            validation_data=(X_val, y_val),
                            callbacks=[checkpointer,csv_logger,reduce_lr,es]) # debug:刪除callbacks中的'es'
        return history

    def predict(self, X_test, y_test):

        if X_test.ndim != 4:
            raise Exception(
                'ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))

        model = self.build()
        model.load_weights(self.weights_dir)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=self.metrics)

        start = time.time()
        y_pred = model.predict(X_test)
        print('shape of y_pred:', y_pred.shape)
        y_pred_argm = np.argmax(y_pred, axis=1)
        # y_test_argm = np.argmax(y_test, axis=1)
        y_test_argm = y_test
        print('shape of y_pred_argm:', y_pred_argm.shape)
        print('shape of y_test:', y_test.shape)
        end = time.time()
        loss, accuracy = model.evaluate(
            x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)

        print(classification_report(y_test_argm, y_pred_argm))
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test_argm, y_pred_argm, average=self.f1_average)
        evaluation = {'loss': loss,
                      'accuracy': accuracy,
                      'f1-score': f1,
                      'prediction_time': end-start}
        Y = {'y_true': y_test_argm,
             'y_pred': y_pred_argm}
        return Y, evaluation


'''
def EEGNet(nb_classes, Chans = 30, Samples = 500, 
             dropoutRate = 0.5, kernLength = 250, F1 = 8, 
             D = 1, F2 = 8, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)
'''

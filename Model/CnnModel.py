#卷积网络模型
from keras import Sequential, Input, Model
from keras.initializers import he_uniform
from keras.layers import Conv2D, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Dense, GlobalAveragePooling1D, \
    Activation, MaxPooling2D, Flatten, LeakyReLU, PReLU, ELU, Masking, Bidirectional, LSTM, add, ZeroPadding2D, merge, \
    ZeroPadding1D, concatenate, Reshape, average, TimeDistributed
from keras.constraints import max_norm
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.regularizers import l1_l2, l2, l1
import numpy as np
np.random.seed(1)


class CNNModel(AttentionWithContext):
    @staticmethod
    def CNN1D(shape):
        model=Sequential()
        model.add(Conv1D(300,3,activation='linear',strides=1,input_shape=shape))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.6))
        model.add(Conv1D(200,3,activation='linear',strides=1))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.6))
        model.add(Conv1D(100,4,activation='linear',strides=1))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(100,activation='linear'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(60, activation='linear'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(Adam(0.001),loss='binary_crossentropy',metrics=['acc'])
        return model

    @staticmethod
    def DNN(shape):
        x,y=shape
        model = Sequential()
        model.add(Dense(100, activation='linear',input_shape=(x*y,)))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(60, activation='linear'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
        return model

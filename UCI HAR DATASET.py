# Import PACKAGES
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime, os
from sklearn import preprocessing
import numpy as np


from sklearn.model_selection import train_test_split


from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Bidirectional, GRU, Dense,Dropout
from tensorflow.keras import optimizers
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
SIGNALS = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
           "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
           "total_acc_x_", "total_acc_y_", "total_acc_z_"]
train_paths = ['C:/Users/admin/PycharmProjects/SONALIProject/UCI HAR Dataset/train/Inertial Signals/' + signal + 'train.txt' for signal in SIGNALS]
test_paths = ['C:/Users/admin/PycharmProjects/SONALIProject/UCI HAR Dataset/test/Inertial Signals/' + signal + 'test.txt' for signal in SIGNALS]
def __load_X(X_signal_paths):
    X_signals = []

    for signal_type_path in X_signal_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))
x_train = __load_X(train_paths)
x_test = __load_X(test_paths)
x_train = __load_X(train_paths)
x_test = __load_X(test_paths)
y_train = np.loadtxt('C:/Users/admin/PycharmProjects/SONALIProject/UCI HAR Dataset/train/y_train.txt',  dtype=np.int32)
y_test = np.loadtxt('C:/Users/admin/PycharmProjects/SONALIProject/UCI HAR Dataset/test/y_test.txt', dtype=np.int32)
print('x_train shape is: ', x_train.shape)
print('x_test shape is: ', x_test.shape)

print('y_train shape is: ', y_train.shape)
print('y_test shape is: ', y_test.shape)

print('Number of classes: ', len(np.unique(y_train)))

print(x_train.shape[1:])
x_train.shape[1]

imput_dim = x_train.shape
input_layer = Input(shape = (imput_dim[1:]))

layer_1 = Bidirectional(GRU(128, activation = 'relu', return_sequences = True,name='encoder_layer_2'))(input_layer)
layer_2 = GRU(50, activation = 'relu', return_sequences = False,name='encoder_layer_3')(layer_1)

layer_3 = RepeatVector(x_train.shape[1],name='repeatVector_layer')(layer_2)

layer_4 = GRU(50, activation = 'relu', return_sequences = True,name='decoder_layer_1')(layer_3)
layer_5 = GRU(128, activation = 'relu', return_sequences = True,name='decoder_layer_2')(layer_4)
output_layer = TimeDistributed(Dense(imput_dim[2]),name='output_layer')(layer_5)

model_GRU = Model(inputs = input_layer , outputs = output_layer)
print(model_GRU.summary())

encoder_GRU = Model(inputs = input_layer , outputs = layer_2)

#
from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=50)
model_GRU.compile(optimizer='adam', loss='mse',metrics=['mae', 'mse'])
encoder_decoder_history = model_GRU.fit(x_train, x_train,
                                              batch_size=512,
                                              epochs=100,
                                              validation_data=(x_test, x_test),
                                              callbacks=[early_stop])
X_train = encoder_GRU(x_train)
X_test = encoder_GRU(x_test)
print('Encoded X_train shape is: ', X_train.shape)
print('Encoded X_test shape is: ', X_test.shape)

from sklearn.ensemble import RandomForestClassifier

def rf(x_train, y_train, n_estimators=300):
 rndforest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
 rndforest.fit(x_train, y_train)
 return rndforest
random_forest_en = rf(X_train,y_train, n_estimators=300)
print("Training accuracy:", random_forest_en.score(X_train, y_train))
print("Validation accuracy", random_forest_en.score(X_test, y_test))

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(model, X, y, class_names, file_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    y_pred = model.predict(X)
    # Compute confusion matrix
    cnf_matrix  = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=1)
    plt.figure(figsize=(18, 16))

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("MATRIX OF CONFUSION")
    else:
        print("MATRIX OF CONFUSION")

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('MATRIX OF CONFUSION')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('ACCIONES (CLASES)')
    plt.xlabel('CLASE PREDICHA')
    plt.tight_layout()
    plt.savefig(file_name+'.png')
    plt.show()
from sklearn.metrics import confusion_matrix

plot_confusion_matrix(random_forest_en, X_test, y_test, class_names=LABELS, file_name='ConfussionMatrix', normalize=True)
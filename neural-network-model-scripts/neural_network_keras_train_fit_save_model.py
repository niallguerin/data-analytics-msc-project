# Author: Niall Guerin
# Change Date: 23.08.2019
# Change Date: 05.10.2022
# Description: Keras-TensorFlow Neural Network updated in October 2022 to
# handle fixes for deprecated TensorFlow calls since 2019. Contains input
# layer, 2 additional layers with dropout added, and 1 output layer using sigmoid
# function. RELU function is used on the other 3 layers. Batch Normalization was
# added back in 2019 along with adam_custom hyperparameter tuning changes. This
# model was slow so more work was needed on the Windows workstation to perform train:test
# exercises and imbalanced dataset was heavily impacting its performance too and inability
# to predict TP answers versus it doing well at least exluding irrelevant wrong answers.
# It could only do well on training and training test set so was overfitting always. I
# choose not to spend more time on it as using this was more an experiment for myself to
# get under the hood of issues faced when working with neural networks, adding/removing layers,
# adding/removing nodes, and hyperparameter tuning tasks. Batch Size and Epochs tended to have
# more impact on performance i.e. even getting it better at True Negative predictions which is
# important for avoiding rubbish being given to the end user in our SO dataset scenario. To
# get this model to perform better, the imbalanced dataset input has to be addressed and the
# existing numeric values in the dataset input needs to be normalized as we can easily see
# poor value ranges on many of the feature columns which we know will impact a neural network.

import numpy as np
import tensorflow as tf
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization

# load the dataset as basis for result set IR system component queries
# dataset = genfromtxt('df_train_so1k.csv', delimiter=',', dtype=float)
# dataset = genfromtxt('df_train_so20k.csv', delimiter=',', dtype=float)
# dataset = genfromtxt('df_train_so_all_answers01.csv', delimiter=',', dtype=float)
# dfcheck = pd.read_csv('df_train_so1k.csv', delimiter=',',)
dataset = genfromtxt('df_train_so_all_answers01.csv', delimiter=',', dtype=float)
# dataset = genfromtxt('df_train_so_all_answers00.csv', delimiter=',', dtype=float)
# dataset = genfromtxt('df_train_so_all_answers41.csv', delimiter=',', dtype=float)
# dataset = genfromtxt('df_train_so_all_answers15_clean_data.csv', delimiter=',', dtype=float)
print(dataset.shape)
# dfcheck100k = dfcheck[:10000]
# print(dfcheck100k.shape)

# using scikit-learn train:test split function
# I am using a slightly clunky piece of code for the split as i find this easier to read
# train and test explicitly defined and split using the sci-kit learn utility library
train, test = train_test_split(dataset, test_size=0.1)

# define the training set: first line is the training set with features, second line is the set with classifier label
Xtrain = train[:,0:23]
ytrain = train[:,23]

# define the test set: first line is the list of features, second line is the label classifier (for test set)
Xtest = test[:,0:23]
ytest = test[:,23]

# define the keras model: base template her from both Keras and my tutorials from machinelearningmaster.com e-textbook
# Web References:
# 1. https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# 2. https://keras.io/layers/about-keras-layers/
# 3. https://keras.io/optimizers/
# 4. For layer added on 22.08.2019, refer to https://keras.io/layers/normalization/

# BACKUP BEST NN result on TEST dataset to date; still poor at predicting TPs
so_nn_model_batch_norm = Sequential()
so_nn_model_batch_norm.add(Dense(60, input_shape=(23,), activation='relu'))
so_nn_model_batch_norm.add(Dense(40, activation='relu'))
so_nn_model_batch_norm.add(Dropout(0.2))
so_nn_model_batch_norm.add(Dense(20, activation='relu'))
so_nn_model_batch_norm.add(Dropout(0.1))
so_nn_model_batch_norm.add(BatchNormalization())
so_nn_model_batch_norm.add(Dense(1, activation='sigmoid'))

# compile the keras model
# web reference: https://keras.io/optimizers/
# set alpha higher than default and set decay at 0.1. Andrew Ng advises not to get distracted playing with decay rate
# so focus on epochs, batch, and alpha. Limited time in last 24 hours, so keeping model basic for experiments; running
# better at predicting TPs versus original NN model which could do nothing useful!
adam_custom = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) - default if I want to roll back
so_nn_model_batch_norm.compile(loss='binary_crossentropy', optimizer=adam_custom, metrics=['accuracy'])

# fit model and refer machinelearningmastery.com for best output format for metrics if needed at fit stage beyond
# output here (for loss and accuracy). Prediction model for result sets will record full metrics
# web reference:
# https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
history = so_nn_model_batch_norm.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), batch_size=50, epochs=20, verbose=0)
# evaluate the model
_, train_acc = so_nn_model_batch_norm.evaluate(Xtrain, ytrain, verbose=0)
_, test_acc = so_nn_model_batch_norm.evaluate(Xtest, ytest, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Note: Don't be fooled by Accuracy results; the other measures are
# critical as the model still cannot get TPs on any tests performed
# ranging from 1k to 3million record dataset inputs and we know the
# dataset format and imbalance of TPs versus TNs is impacting it.
# It only does OK on training dataset, which is no use as it cannot
# be generalized for best answer prediction in our application use case.

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='Training Set')
pyplot.plot(history.history['val_loss'], label='Test Set')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='Training Set')
pyplot.plot(history.history['val_accuracy'], label='Test Set')
pyplot.legend()
pyplot.show()

# save the model - modify this so as to avoid over-writing any prior model I was satisfied with in testing
so_nn_model_batch_norm.save('nn_model_so_100k.h5')

# print the model summary so we have a record of the layers/setup for records
print(so_nn_model_batch_norm.summary())
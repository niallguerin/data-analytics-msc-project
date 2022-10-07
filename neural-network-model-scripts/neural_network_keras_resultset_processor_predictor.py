# Author: Niall Guerin
# Change Date: 23.08.2019
# Change Date: 05.10.2022
# Description:
# Workflow:
# 1. Load the Keras Neural Network model
# 2. Parse resultsets from IR input query result output
# 3. Make prediction of best answer against that resultset and provide web snippet in CLI

import numpy as np
from numpy import genfromtxt
from keras.models import load_model
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# load the IR resultset dataset

# configure list of filenames so we can run experiments from this point only in workflow and not need E2E retest from
# IR system component; helps for quick model tuning and re-deploy of model updates
# the input file represents precision@k and query test case number
infile_q1_k15 = "df_resultset_answers00_q1_k15.csv"
infile_q1_k10 = "df_resultset_answers00_q1_k10.csv"
infile_q1_k5 = "df_resultset_answers00_q1_k5.csv"

infile_q2_k15 = "df_resultset_answers00_q2_k15.csv"
infile_q2_k10 = "df_resultset_answers00_q2_k10.csv"
infile_q2_k5 = "df_resultset_answers00_q2_k5.csv"

infile_q3_k15 = "df_resultset_answers00_q3_k15.csv"
infile_q3_k10 = "df_resultset_answers00_q3_k10.csv"
infile_q3_k5 = "df_resultset_answers00_q3_k5.csv"

infile_q4_k15 = "df_resultset_answers00_q4_k15.csv"
infile_q4_k10 = "df_resultset_answers00_q4_k10.csv"
infile_q4_k5 = "df_resultset_answers00_q4_k5.csv"

infile_q5_k15 = "df_resultset_answers00_q5_k15.csv"
infile_q5_k10 = "df_resultset_answers00_q5_k10.csv"
infile_q5_k5 = "df_resultset_answers00_q5_k5.csv"

dataset = genfromtxt(infile_q1_k5, delimiter=',', dtype=float)
# split into input features and output classification label i.e. solution 0 or 1 in our stack overflow use case
X_so_result_set = dataset[:, 0:23]
y_so_result_set = dataset[:, 23]

# constant for looping over predictions
range_counter = 10

# load the previously saved Batch Normalized model (extra layer added on 22.08.2019)
so_nn_model = load_model('nn_model_so_100k.h5')

# make class predictions with the model on test data i.e. the resultset from the IR system filtered search results
# OLD CODE - REMOVED THIS LINE predictions
# so_nn_model.predict_classes(X_so_result_set)
# fix required due to my library updates 03/10/2022 to allow old keras/tf nn to run.
# https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
predictions = so_nn_model.predict(X_so_result_set)

# evaluate the model
scores = so_nn_model.evaluate(X_so_result_set, y_so_result_set)
print("NN model metrics: ", (so_nn_model.metrics_names[1], scores[1] * 100))

# add the metrics using following as primary resources for pattern of measurements:
# 1. https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# 3. My case studies python assignments with evaluation reports

# predict probabilities for stack overflow result set from IR system component
so_result_set_probabilities = so_nn_model.predict(X_so_result_set, verbose=0)
# predict classes
so_result_set_classes = so_nn_model.predict(X_so_result_set)
so_result_set_classes = np.argmax(so_result_set_classes, axis=1)
# print("numpy predicted classes check for column data type")
# print(so_result_set_classes)

# modify dimensions of probabilities and class arrays (refer to original machinelearningmastery.com template from tutorial)
so_result_set_probabilities = so_result_set_probabilities[:, 0]
so_result_set_classes = so_result_set_classes[:, 0]

# calculate required standard measures set out under evaluation in thesis
auc = roc_auc_score(y_so_result_set, so_result_set_probabilities)
print('AUC:', auc)

# calculate roc curve using true positive and false positive values
fpr, tpr, thresholds = roc_curve(y_so_result_set, so_result_set_probabilities)

# my plot I used from older model broke here, so using this metrics output template from e-book
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot roc curve for so nn model predictions on IR result set
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()

# print accuracy
# FIX UPDATE in October 2022: see https://stackoverflow.com/questions/38015181/accuracy-score-valueerror-cant-handle-mix-of-binary-and-continuous-target
accuracy = accuracy_score(y_so_result_set, so_result_set_classes)
print('Accuracy: ', accuracy)

# precision:
precision = precision_score(y_so_result_set, so_result_set_classes)
print('Precision: ', precision)
# recall: see thesis for formula
recall = recall_score(y_so_result_set, so_result_set_classes)
print('Recall: ', recall)
# f1 score
f1 = f1_score(y_so_result_set, so_result_set_classes)
print('F1 score: ', f1)

# Get auroc value and will need it for plotting the curve graph
roc_auc_score = roc_auc_score(y_so_result_set, predictions)
print("AUROC:", roc_auc_score)

# print the model summary so we have a record of the layers/setup for records
print(so_nn_model.summary())

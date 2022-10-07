# Author: Niall Guerin
# Change Date: 23.08.2019
# Change Date: 05.10.2022
# Description: XGBoost implementation using scikit-learn to provide best answer prediction. This provides
# save and load model functions. Implemented model save/load fixes in October 2022 to
# handle deprecated method calls since 2019.

from numpy import loadtxt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# load data
dataset = loadtxt('df_train_so_all_answers01.csv', delimiter=",")

# split data into X and y: keeping X, y format this time per all the documentation and sites
X = dataset[:,0:23]
Y = dataset[:,23]

# split data into train and test sets and set seed for reproducibility
seed = 7
test_size = 0.33
# syntax below based on sklearn api documentation examples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# construct model using scikit-learn XGB classifier
so_xgb_model = XGBClassifier()
so_xgb_model.fit(X_train, y_train)

# make predictions for so test dataset
y_pred = so_xgb_model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate prediction accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", (accuracy * 100.0))

# summarize the first N cases
for i in range(len(predictions)):
	print("Prediction is: ", predictions[i])
	print("Test value expected is: ", y_test[i])

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", (accuracy * 100.0))

# calculate f-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
f_score_label = "F-Score Measure is: "
f_score = fbeta_score(y_test, predictions, average='macro', beta=0.5)
print(f_score_label, f_score)

# print confusion matrix
confusion_matrix_header = "Confusion Matrix: "
print(confusion_matrix_header)
cm = confusion_matrix(y_test, predictions)
print(confusion_matrix(y_test, predictions))

# This tablular printout contains full classification report in human-readable form on CLI console
print("Classification Report: ")
print(classification_report(y_test, predictions))

# get confusion matrix values explicitly and use for reporting table in conjunction with cm visualization
true_negative = cm[0][0]
print("True Negative", true_negative)
false_negative = cm[1][0]
print("False Negative", false_negative)
true_positive = cm[1][1]
print("True Positive", true_positive)
false_positive = cm[0][1]
print("False Positive", false_positive)

# Show confusion matrix
pyplot.matshow(cm)
pyplot.title('XGBoost ML Model: Confusion Matrix')
pyplot.colorbar()
pyplot.ylabel('True')
pyplot.xlabel('Predicted')
pyplot.show()

# fpr, tpr = scikit learn standard format for false postive rate, true positive rate
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# create the roc curve plot: use benalexkeen.com blog as main reference, customize and using scikit learn samples
# from my applied case studies projects in semester 2
pyplot.figure()
pyplot.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC Curve for XGBoost ML Model')
pyplot.legend(loc="lower right")
pyplot.show()

# save the SO XGB ML model to file system
# Updated 3/10/2022: save/load not working with code based on new api version installed for review test
# Modifying code to support new save/load syntax as advised by compiler, API documentation.

# save models in json and text format
so_xgb_model.save_model("so_xgb_model.json")
# save in text format
so_xgb_model.save_model("so_xgb_model.txt")
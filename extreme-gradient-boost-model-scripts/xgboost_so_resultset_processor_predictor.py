# Author: Niall Guerin
# Change Date: 23.08.2019
# Change Date: 05.10.2022
# Description: XGBoost implementation using scikit-learn to provide best answer prediction.
# Workflow:
# 1. Load the XGBoost model
# 2. Parse resultsets from IR input query result output
# 3. Make prediction of best answer against that resultset and provide web snippet in CLI

from numpy import loadtxt
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd

# filename constant for displaying snippets (make sure this is correct or snippet will be garbage from irrelevant corpus)

source_file = "data/all_answers00.csv"

# configure list of filenames so we can run experiments from this point only in workflow and not need E2E retest
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

# load data
dataset = loadtxt(infile_q4_k5, delimiter=",")

# split data into X and y
X_test = dataset[:,0:23]
y_test = dataset[:,23]

# load last saved so xgb model from file
# filename = 'old-api-version/xgboost_so_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
loaded_model = XGBClassifier()
loaded_model.load_model('so_xgb_model.json')

result = loaded_model.score(X_test, y_test)

# make predictions for test data
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]

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

# print("INSPECT Confusion Matrix: ")
# print("TRUE values: ")
# print(y_test)
# print("PREDICTED values: ")
# print(y_pred)
# print("DISPLAY CM")
# print(cm)

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
plt.matshow(cm)
plt.title('XGBoost ML Model: Confusion Matrix')
plt.colorbar()
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Get auroc value and will need it for plotting the curve graph
roc_auc_score = roc_auc_score(y_test, predictions)
print("AUROC:", roc_auc_score)

# plot roc curve graph per guidelines in web reference samples
# Web References:
# http://benalexkeen.com/scoring-classifier-models-using-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# fpr, tpr = scikit learn standard format for false postive rate, true positive rate
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# create the roc curve plot: use benalexkeen.com blog as main reference, customize and using scikit learn samples
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost ML Model')
plt.legend(loc="lower right")
plt.show()

# # handle snippet extraction: this is flaky and not changing at this time. Hacked as quick web link utility
best_predicted_answer_ids = []
correlation_list = []
range_counter = len(predictions)
for i in range(range_counter):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test[i]))
	if (predictions[i] == 1):
		print("Best Answer Predicted!")
		print('Answer Details is =>', X_test[i][0])
		# append the question_id to the correlation list - filter on this id from all_answers master dataset later
		correlation_list.append(int(X_test[i][5]))
		best_predicted_answer_ids.append(int(X_test[i][0]))

# using original stackoverflow url mapper from June work, output the question id as a url for user
# https://stackoverflow.com/questions/question_id/
answer_list = []
for index in range(len(best_predicted_answer_ids)):
	url_answer_base = 'https://stackoverflow.com/questions/'
	ans_question_id = str(best_predicted_answer_ids[index])
	url_answer = url_answer_base + ans_question_id
	answer_list.append(url_answer)

print("Final Answer Set Size: ", len(answer_list))
print(answer_list)

# give user the snippet of text from the specific answer within the thread of answers
# define feature names for reading the staging table csv for all_answers export file and columns
input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

# load the raw staging all_answers_N = size export dump
df_collection = pd.read_csv(source_file, sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')
#print(df_collection.head())
if len(answer_list) == 0:
	print("XGBoost was unable to identify a best answer prediction. It only predicted true negative values.")
else:
	for value in correlation_list:
		print("Correlation ID : ", value)

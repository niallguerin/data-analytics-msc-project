# handle age and time based meta features

# Web Reference: https://github.com/collab-uniba/emse_best-answer-prediction/tree/master/input

# Author: Niall Guerin

# Date Created: 30.06.2019
# Date Updated: 23.08.2019
# Any source code referencing an external pattern or resource is cited at line level to acknowledge that source

# import pandas for data set processing
import pandas as pd
import numpy as np

# begin function library
def calculate_time_difference(input_df):
	# format columns to correct data types
	input_df['question_date'] = pd.to_datetime(input_df['question_date'])
	input_df['answer_date'] = pd.to_datetime(input_df['answer_date'])
	# print(input_df['question_date'])
	# print(input_df['answer_date'])

	# calculate age: this calculates the difference between the original question post date (refer to original Calefato paper)
	# and the answer date
	input_df['time_difference'] = (
			input_df['answer_date'] - input_df['question_date']).dt.seconds
	input_df['time_difference'] = pd.to_timedelta(input_df['time_difference'], unit='s')
	input_df['time_difference'] = input_df['time_difference'] / np.timedelta64(1, 's')

	# Reminder to Self: Cross-check a sample file with the ones in R Caret from Calefato to confirm the time differences match
	# as it is possible to correlate the records with the right question ID in mysql staging Posts table

	# rename the existing column to required feature engineering header format
	input_df.rename(columns={'answer_date': 'date_time'}, inplace=True)
	# print(input_df['time_difference'])
	return input_df
# end function library

# define feature names for reading the staging table csv for all_answers export file and columns
input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

# define features columns to drop and customize to allow for retention of question_id in our case as I want this for simple snippet extraction
# later in the workflow. This is a requirement from my own project as this project is building on an earlier
# IR system project so the snippet should provide question-answer summary and context for an end user and a link
# to the SO live website question-answer once the ML make a best answer prediction (even if prediction is right or wrong)
drop_column_targets = ['question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'question_date', 'answer_id',
					  'answers_count', 'answer_body', 'answers_count']

# read sample files and use nrows for mini test cases with new data files
# REMINDER: the second file is the held out 100k dataset for train and test so within that holdout 100k chunk
# the ml models during training will hold out a further portion for testing
# df = pd.read_csv("data/all_answers.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\', nrows=20000)
# df = pd.read_csv("data/all_answers01.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')

# print("DataFrame Size from Large File Read")
# print(df.shape)

# calculate the time and time difference features
df_updated = calculate_time_difference(df)

# rename the upvote-downvote column to upvotes: posts staging table already has the precalculated value per row so don't code it from scratch!
df_updated.rename(columns={'answer_upvotescore': 'upvotes'}, inplace=True)

# drop column names we no longer need
df_updated = df_updated.drop(columns=drop_column_targets)
print("Updated DataFrame Size")
print(df_updated.shape)

# export meta features to csv
df_updated.to_csv("data/df_meta_features.csv", encoding='utf-8', index=False)
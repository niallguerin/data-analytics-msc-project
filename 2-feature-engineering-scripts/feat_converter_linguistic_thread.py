# Description: Feature Converter Script for Linguistic Feature Engineering step for machine learning model training.
# It is used for transforming the output of all_answers*.csv from SO dataset master staging to numeric feature formats
# described at this specification page.
# Web Reference: https://github.com/collab-uniba/emse_best-answer-prediction/tree/master/input

# Author: Niall Guerin

# Date Created: 30.06.2019
# Date Updated: 16.07.2019
# Any source code referencing an external pattern or resource is cited at line level to acknowledge that source

# import pandas for data set processing and filtering
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# begin function library
# function library to carry out primary feature engineering tasks: function names should self-describe their task
def get_length_of_characters(answer_text):
	return len(answer_text)

def get_word_count(answer_text):
	word_tokens = word_tokenize(answer_text)
	word_count = len(word_tokens)
	return word_count

def get_number_of_sentences(answer_text):
	# use value accessor because otherwise answer_text is a pandas dataframe structure and nltk will throw exception
	sentences = answer_text
	sent_tokens = sent_tokenize(sentences)
	return len(sent_tokens)

def get_longest_sentence(answer_text):
	sent_tokens = sent_tokenize(answer_text)
	return len(max(sent_tokens))

def get_avg_words_per_sentence(answer_text):
	# get average words per sentence
	total_words = 0
	num_sentences = 0
	sent_tokens = sent_tokenize(answer_text)
	# loop over the sentences extracted via nltk from the answer_body
	for idx, sentence in enumerate(sent_tokens):
		# tokenize the sentences into words
		word_tokens = word_tokenize(sentence)
		# increment the word total and sentence total and compute the average
		total_words += len(word_tokens)
		num_sentences += 1
	avg_words_per_sent = total_words / num_sentences
	return avg_words_per_sent

def get_avg_characters_per_word(answer_text):
	# get average characters per word
	total_chars = 0
	word_tokens = word_tokenize(answer_text)
	num_words = len(word_tokens)
	for index, word in enumerate(word_tokens):
		total_chars += len(word)
		avg_characters_per_word = total_chars / num_words
	return avg_characters_per_word

def has_hyperlinks(answer_text):
	# The following regex pattern is based on this stack overflow thread as this can be tricky per thread comments but this one was sufficient for my testing
	# regex pattern source Web References: https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
	# The below regex string is amalgamated from an answer in one of the threads on this topic similar to above as I re-used an existing pattern
	url_pattern = re.compile(
		"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")
	is_match = bool(url_pattern.search(answer_text))
	return is_match

# perform linguistic feature conversions
def perform_linguistic_feature_conversion(df):

	# loop over dataframe in pandas using the apply function rather than loop in original version
	df['len'] = df['answer_body'].apply(get_length_of_characters)
	df['wordcount'] = df['answer_body'].apply(get_word_count)
	df['avg_chars_per_word'] = df['answer_body'].apply(get_avg_characters_per_word)
	df['sentences'] = df['answer_body'].apply(get_number_of_sentences)
	df['avg_words_per_sentence'] = df['answer_body'].apply(get_avg_words_per_sentence)
	df['longest_sentence'] = df['answer_body'].apply(get_longest_sentence)
	df['has_links'] = df['answer_body'].apply(has_hyperlinks)

	return df
# end function library

# define feature names for reading the staging table csv for all_answers export file and columns
input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

# define features columns to drop and customize to allow for retention of question_id/aka post id in the research paper
drop_column_targets = ['question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'question_date',
					  'answer_date', 'answer_upvotescore', 'answer_body']

# read full all_answers dataset; use the nrows attribute template line to use the constants for number of records if you need to run mini tests
# our dataset is static historical SO data not live streaming dataset
# the following were used for a :
# multi-million record file (3 million) records
# a 100k test dataset containing the questions and answers from 4 our target IR system queries and ML prediction query-answer test cases
# a 100k holdout train and test set that contains data not relevant to any 5 of our target IR system or ML prediction query-answer test cases
# df = pd.read_csv("data/all_answers.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\', nrows=20000)
# df = pd.read_csv("data/all_answers01.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')
# df = pd.read_csv("data/all_answers15.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')

# utility debug print statements for quick validation and eda
# print(len(features_linguistic_sample.columns))
# print(df_updated.columns)
df_updated = perform_linguistic_feature_conversion(df)

# drop column names we no longer need
df_updated = df_updated.drop(columns=drop_column_targets)

# export linguistic_features to csv
df_updated.to_csv("data/df_linguistic_thread_features.csv", encoding='utf-8', index=False)

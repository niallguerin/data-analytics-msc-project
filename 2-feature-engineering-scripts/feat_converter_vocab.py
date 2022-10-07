# calculate the flesh-kincaid grade for readability
# update the log-likelihood as dropped from feature processing and set to 0

# Web Reference: https://github.com/collab-uniba/emse_best-answer-prediction/tree/master/input

# Author: Niall Guerin

# Date Created: 30.06.2019
# Date Updated: 23.08.2019
# Any source code referencing an external pattern or resource is cited at line level to acknowledge that source

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
# use built-in F-K functions from TextStat API
import textstat
import pandas as pd
import nltk
nltk.download('punkt')

# begin function library

# construct the so_dataset corpus from the answers staging csv
def createCorpus(df):
	# filter out only the answer_body here per paper specification (see external site at top of this file)
	target_columns = ['question_id', 'answer_body']
	df_updated = pd.DataFrame(df, columns=target_columns)

	return df_updated

# preprocess the corpus, remove stopwords
def preprocessCorpus(df):

	# define stopwords: uses nltk default. This whole area is a minefield for the SO dataset so this is an area for huge improvement overall in
	# future builds to optimize how we parse the English text in the dataset
	stop_words = set(stopwords.words('english'))
	# print(stop_words)

	# collection for corpus
	so_corpus = set()

	# loop over raw corpus and preprocess tokenizing by sentence in answer_body, then word_tokenize on sentence, and then stopword removal
	for answer_index, answer_body in enumerate(range(0, len(df))):
		answer_text = df.iloc[answer_index, :]['answer_body']
		sent_tokens = sent_tokenize(answer_text)
		for idx, sentence in enumerate(sent_tokens):
			word_tokens = word_tokenize(sentence)
			# print(word_tokens)
			for word in word_tokens:
				if word not in stop_words:
					so_corpus.add(word)
	return so_corpus

# utility function to print vocabulary to console
def display_vocab(df_input):
	for index, vocab_item in enumerate(df_input):
		print("Index: ", index, "Vocabulary Item: ", vocab_item)
	return

def get_flesch_kincaid_grade(answer_text):
	# get the F-K grade via call to TextStat function
	flesch_kindcade_grade = textstat.flesch_kincaid_grade(answer_text)
	# print("Flesch Kincaid Grade", flesch_kindcade_grade)

	return flesch_kindcade_grade

# end function library

# read the all_answers.csv sample file
input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

# read all file
df = pd.read_csv("data/all_answers01.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')
# df = pd.read_csv("data/all_answers15.csv", sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')

# process the dataframe and perform vocabulary feature extraction from the corpus
df_vocab = createCorpus(df)
df_vocab = preprocessCorpus(df_vocab)

# utility function to display vocabulary
# display_vocab(df_vocab)

# set log-likelihood to 0 for the dataframe. We do not care about it here anymore as it slows down feature engineering step too much on laptop
# and it has not significant value according to original paper authors
df['loglikelihood'] = 0

# get flesch-kincaid grade over answer_body column
df['F-K'] = df['answer_body'].apply(get_flesch_kincaid_grade)

# define features columns to drop and customize to allow for retention of question_id/aka post id in the research paper
drop_column_targets = ['question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'question_date', 'answer_id',
					  'answer_date', 'answer_upvotescore', 'answer_body', 'answers_count']

# drop column names we no longer need
df_test_sample = df.drop(columns=drop_column_targets)

# export vocabulary features to csv
df_test_sample.to_csv("data/df_vocabulary_features.csv", encoding='utf-8', index=False)


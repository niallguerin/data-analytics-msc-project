# merger utility script to merge dataframes from individual preprocessor script runs

# Author: Niall Guerin

# Date Created: 30.06.2019
# Date Updated: 23.08.2019
# Any source code referencing an external pattern or resource is cited at line level to acknowledge that source

import pandas as pd
# read individual feature engineering csv input files: linguistic features, meta-features, vocabulary-features
df1 = pd.read_csv("data/df_linguistic_thread_features.csv", sep=",", engine='c', header=0, escapechar='\\')

df2 = pd.read_csv("data/df_meta_features.csv", sep=",", engine='c', header=0, escapechar='\\')
df2 = df2.rename(columns={'question_id': 'qid'})

df3 = pd.read_csv("data/df_vocabulary_features.csv", sep=",", engine='c', header=0, escapechar='\\')
df3 = df3.rename(columns={'question_id': 'qid'})

# append columns only and merge on question_id: keep print utility statements in case you end up with garbage dataframe output breaking ml preprocessor
df_merged_1 = pd.concat([df1, df2], axis=1)
# print(df_merged_1)
# print(df_merged_1.columns)

df_merged_1 = df_merged_1.drop(columns='qid', axis=1)
# print(df_merged_1.columns)
df_merged_2 = pd.concat([df_merged_1, df3], axis=1)
# print(df_merged_2.columns)
df_merged_2 = df_merged_2.drop(columns='qid', axis=1)
# print(df_merged_2.columns)

# rank features listed in calefato paper for ranking only
# Web Reference: https://github.com/collab-uniba/emse_best-answer-prediction/tree/master/input
df_merged_2['len_rank'] = df_merged_2.groupby('question_id')['len'].rank(ascending=False)
df_merged_2['wordcount_rank'] = df_merged_2.groupby('question_id')['wordcount'].rank(ascending=False)
df_merged_2['avg_chars_per_word_rank'] = df_merged_2.groupby('question_id')['avg_chars_per_word'].rank(ascending=False)
df_merged_2['sentences_rank'] = df_merged_2.groupby('question_id')['sentences'].rank(ascending=False)
df_merged_2['avg_words_per_sentence_rank'] = df_merged_2.groupby('question_id')['avg_words_per_sentence'].rank(ascending=False)
df_merged_2['longest_sentence_rank'] = df_merged_2.groupby('question_id')['longest_sentence'].rank(ascending=False)
df_merged_2['time_difference_rank'] = df_merged_2.groupby('question_id')['time_difference'].rank(ascending=True)
# per Ghotsis et al. the closer to 0 the higher the rank as closer to 0 means higher match to so dataset probability as documented in Calefato's paper
df_merged_2['loglikelihood_ascending_rank'] = (df_merged_2['loglikelihood'] < 0.5).astype(int)
df_merged_2['F-K_ascending_rank'] = df_merged_2.groupby('question_id')['F-K'].rank(ascending=False)
df_merged_2['upvotes_rank'] = df_merged_2.groupby('question_id')['upvotes'].rank(ascending=False)

# NB: apply the classification label used by the machine learning models for training
def get_solution_label(input):
	# upvotes_rank 1 are the optimal answers; anything else is not considered a best solution
	if input == 1:
		solution_label = True
	else:
		solution_label = False

	return solution_label

# ************************************
# VERY NB:!!! Get the classification label: the RANK of the upvotes is critical as this becomes equivalent to solution classifier label!!!
df_merged_2['solution'] = df_merged_2['upvotes_rank'].apply(get_solution_label)
# ************************************

# mask user data as we don't want this displayed
def set_answer_uid(input):
	uid_value = input + 1
	answer_uid = str(input) + '.' + str(uid_value)
	return answer_uid

# add column answer_uid
df_merged_2['answer_uid'] = df_merged_2['question_id'].apply(set_answer_uid)

# rearrange the column values to match the original R Caret ml input examples from research paper so my workflow is
# in sync with their original feature format
df_output = df_merged_2[['question_id', 'answers_count', 'answer_uid', 'date_time', 'time_difference', 'time_difference_rank', 'solution', 'len', 'len_rank', 'wordcount', 'wordcount_rank', 'avg_chars_per_word', 'avg_chars_per_word_rank', 'sentences', 'sentences_rank', 'avg_words_per_sentence', 'avg_words_per_sentence_rank', 'longest_sentence', 'longest_sentence_rank', 'loglikelihood', 'loglikelihood_ascending_rank', 'F-K', 'F-K_ascending_rank', 'upvotes', 'upvotes_rank', 'has_links']]

# configure decimal rounding to 2 decimal places on target columns
df_output = df_output.round(2)

# sort time columns to match research paper sample input formats in their csv files and keep it consistent
df_output = df_output.sort_values(['question_id', 'time_difference', 'time_difference_rank'], ascending=[True, True, True])

# below section differs from 2-feature-engineering-scripts syntax for this script only
# We are emulating an IR workflow in this script so we need to calculate precision of
# information retrieval component itself for different resultsets against the original input
# query at results = 5, results = 10, and results = 15. In the live workflow the ML model will
# perform a prediction BUT it has a dependency on the OUTPUT of this module so if this module passes
# a rubbish resultset from the corpus after the input query test case, the ML model even if it gives a
# an accurate best answer prediction (wrong result anyway) according to itself, it will be wrong so
# this IR module needs to pass RELEVANT answerset to ML model and from that the ML model will then
# aim to isolate the single best answer from the filtered resultset.

# configure outfile names to allow quick repeat tests if required downstream in ml model without re-running workflow
outfile_q1_k15 = "so_qs_resultset_answers00_q1_k15.csv"
outfile_q1_k10 = "so_qs_resultset_answers00_q1_k10.csv"
outfile_q1_k5 = "so_qs_resultset_answers00_q1_k5.csv"

outfile_q2_k15 = "so_qs_resultset_answers00_q2_k15.csv"
outfile_q2_k10 = "so_qs_resultset_answers00_q2_k10.csv"
outfile_q2_k5 = "so_qs_resultset_answers00_q2_k5.csv"

outfile_q3_k15 = "so_qs_resultset_answers00_q3_k15.csv"
outfile_q3_k10 = "so_qs_resultset_answers00_q3_k10.csv"
outfile_q3_k5 = "so_qs_resultset_answers00_q3_k5.csv"

outfile_q4_k15 = "so_qs_resultset_answers00_q4_k15.csv"
outfile_q4_k10 = "so_qs_resultset_answers00_q4_k10.csv"
outfile_q4_k5 = "so_qs_resultset_answers00_q4_k5.csv"

outfile_q5_k15 = "so_qs_resultset_answers00_q5_k15.csv"
outfile_q5_k10 = "so_qs_resultset_answers00_q5_k10.csv"
outfile_q5_k5 = "so_qs_resultset_answers00_q5_k5.csv"

df_output.to_csv(outfile_q1_k5, encoding='utf-8', index=False)
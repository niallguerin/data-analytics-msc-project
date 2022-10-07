# Mapper script to map resultset ranking of posts from IR system component to the input for FE scripts
import pandas as pd

# filename constant: change if testing a new corpus file e.g. all_answers9056.csv for question5
filename = "data/all_answers00.csv"

# function to fetch related question_ids to the one's ranked so ml model predicts snippet from a set not just one. This
# is important as otherwise you can pass a single record/question-answer from SO dataset from IR system component and just
# get luck (or not) at ml model prediction step.
#
# The model is designed to predict the best answer from a thread of answers
# with the SAME question, or at least that is the model training goal so we
# ultimately fetch a precise snippet and avoid big lists of results. Here we
# still need to avoid a single lucky record being passed implying the model
# is better than it is in terms of fitting to the problem domain
def fetch_related_question_ids(source, results):
	print(results)
	# print(source.head())

	# fetch the source question ids from initial result set
	filtered_source = source.loc[source.index.isin(results), :]
	# print(source)

	# now drop the duplicate values so for example if I have [1,2,3,4,5 and end up with 11111,22222,3333] just get the single IDs
	# so result = [1,2,3]
	source_qids = filtered_source.drop_duplicates('question_id')
	source_qids_list = source_qids['question_id'].tolist()
	print(source_qids_list)

	# now fetch the related row indices with MATCHING question_ids
	updated_results = source[source['question_id'].isin(source_qids_list)]
	print(updated_results)

	return updated_results

def fetch_source_corpus(file):
	# define feature names for reading the staging table csv for all_answers export file and columns
	input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

	df_collection = pd.read_csv(file, sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')
	# print(df_collection.head())

	return df_collection

# read the result set ids based on precision@K test case value you are working on
result_list_for_mapper_k15 = "resultlist_for_mapper_k15.csv"
result_list_for_mapper_k10 = "resultlist_for_mapper_k10.csv"
result_list_for_mapper_k5 = "resultlist_for_mapper_k5.csv"

df_resultset_input = pd.read_csv(result_list_for_mapper_k5, escapechar='\\')

# convert the required input to a list
result_set_list = df_resultset_input['record_ids'].tolist()

# get the source file
df_source = fetch_source_corpus(filename)

# filter out the records based on ir client result set
df_resultset = fetch_related_question_ids(df_source, result_set_list)

# export to csv
df_resultset.to_csv("fe_conversion_dynamic/df_resultset.csv", encoding='utf-8', index=False, header=False)
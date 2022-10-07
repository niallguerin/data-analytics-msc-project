# load the dataset, extract the corpus, extract the dictionary using gensim, save the dictionary
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora
from gensim.utils import simple_preprocess
from six import iteritems
from nltk.corpus import stopwords
import string

# construct dataset from the all_answers.csv staging table .csv source file and store for now as .csv with just the text
# for gensim

# filename constant
filename = "data/all_answers00.csv"

# define feature names for reading the staging table csv for all_answers export file and columns
input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

# read full all_answers dataset and sample a subset for search resultset construction: 100 to 500K records
small = 100
medium = 20000
large = 100000
very_large = 500000

# read control input file from file splitter operation from complete so_dataset (67GB file) / 27 million answers
# input done using control drops of 100K files for IR system component to allow it run fast on OSX laptop using gensim query similarity calls
# ml model train:test will be kept separate to the drop files used in experiments so predictions should always be on a hold-out set
# the use of the 100K split files allows more interesting queries based on my thesis / coursework queries versus reverse engineered or "staged" queries
df_collection = pd.read_csv(filename, sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')
# print(df_collection.head())

# configure drop column targets
drop_column_targets = ['accepted_answer_id', 'question_date', 'answer_date', 'answer_upvotescore', 'answers_count', 'answer_id']

# drop column names we no longer need
df_collection = df_collection.drop(columns=drop_column_targets)

# merge the documents in collection into single document column for search resultset step
df_collection['document'] = df_collection['question_title'].astype(str) + '_' + df_collection['question_body'].astype(str) + '_' + df_collection['answer_body']
# print(df_collection.columns)
# print(df_collection.document)

# export just the text Q+A string data to file
df_collection['document'].to_csv("data/corpus_so.txt", encoding='utf-8', index=False)

# now set up gensim processing for corpus and dictionary
# collect statistics about all tokens
so_dictionary = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open('data/corpus_so.txt', encoding='utf8'))

# use nltk stopwords
stop_words_list = set(stopwords.words('english'))

# add punctuation characters as a lot of them in stack overflow answer posts
for character in string.punctuation:
	stop_words_list.add(character)

# remove stop words and later words that appear only once; below pattern is re-used from initial gensim tutorials
# Web Reference: https://radimrehurek.com/gensim/tut1.html
stop_ids = [
	so_dictionary.token2id[stopword]
	for stopword in stop_words_list
		if stopword in so_dictionary.token2id
]

# filter out tokens where token occurs only once: stop_ids and once_ids will be removed later
# this has never been tuned and the whole parsing can be optimized. Time constraints meant I had to focus on other
# parts of the workflow to wire up the system end-to-end
single_occurrence_token_ids = [tokenid for tokenid, docfreq in iteritems(so_dictionary.dfs) if docfreq == 1]

# remove stop words and words that appear only once per tutorial from gensim
so_dictionary.filter_tokens(stop_ids + single_occurrence_token_ids)
# so_dictionary.filter_tokens(stop_ids)

# remove gaps in id sequence after words that were removed per tutorial: gensim documents and GitHub issue tracker
# on their repository regularly reminds people not to forget to call compactify if using stopword/other token removal!
so_dictionary.compactify()
print(so_dictionary)

# persist the dictionary for stack overflow dataset to disk
so_dictionary.save('so_dictionary.dict')
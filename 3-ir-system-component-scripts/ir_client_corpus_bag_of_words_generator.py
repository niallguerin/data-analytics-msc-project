# load the dataset, extract the corpus, extract the bag of words corpus using gensim, save the bag of words corpus

from gensim import corpora
from gensim.utils import simple_preprocess
import pandas as pd

# gensim tutorials keep logging on and it is very useful for tracking details like matrix size, feature length, and
# matrix density, so logging is enabled primarily for benefit of gensim tracing for all the ir_client* files.
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# construct dataset from the all_answers*.csv staging table .csv source file and store for now as .csv with just the text
# for gensim

# filename constant
filename = "data/all_answers00.csv"

# define feature names for reading the staging table csv for all_answers export file and columns
input_feature_names = ['question_id', 'question_body', 'question_title', 'question_tags', 'accepted_answer_id', 'answers_count',
			   'question_date', 'answer_id', 'answer_date', 'answer_upvotescore', 'answer_body']

# read full all_answers dataset and sample a subset for search resultset construction: 100 records to 500K. Thesis uses 100K
# for measurements
small = 100
medium = 20000
large = 100000
very_large = 500000

# on a separate dell xps workstation with 4 times the RAM of my macbook, we can process
# files up to 3 million records and run ML training models in minutes on them. ML models
# performance was similar at 100K and 3million record train:test sets

# read full file
df_collection = pd.read_csv(filename, sep=";", engine='c', header=None, names=input_feature_names, escapechar='\\')
# print(df_search.head())

drop_column_targets = ['accepted_answer_id', 'question_date', 'answer_date', 'answer_upvotescore', 'answers_count', 'answer_id']

# drop column names we no longer need
df_collection = df_collection.drop(columns=drop_column_targets)

# merge the documents in collection into single document column for search resultset step
df_collection['document'] = df_collection['question_title'].astype(str) + '_' + df_collection['question_body'].astype(str) + '_' + df_collection['answer_body']
print(df_collection.columns)
print(df_collection.document)

# tester container as list
# corpus_list = df_collection['document'].tolist()
# print(corpus_list[0])
# output the pandas dataframe column to a text file so we only have text information we need.
# This is also important later for generating answer snippets
df_collection['document'].to_csv("data/corpus_so.txt", encoding='utf-8', index=False)

# load the so dataset dictionary
so_dictionary = corpora.Dictionary.load('so_dictionary.dict')
# print(so_dictionary)

# construct the corpus which is what we will use to perform query similarity queries against and tf-idf tasks
# the following class is a simple object based on Gensim tutorial sample setup templates
# Web Reference: https://radimrehurek.com/gensim/tutorial.html and specifically the class template is based on this
# https://radimrehurek.com/gensim/tut1.html#corpus-streaming-one-document-at-a-time
# Below pattern for corpus streaming which does not overwhelm system memory when I use larger .csv corpus input files
class StackOverflowCorpus(object):
	def __iter__(self):
		for line in open('data/corpus_so.txt', encoding='utf8'):
			# uses gensim simple preprocessor function = 1 document per csv line item and whitespace separator
			document_tokens = simple_preprocess(line, deacc=True)
			so_bow = so_dictionary.doc2bow(document_tokens)

			# return the bag of words/gensim tf-idf matrix
			yield so_bow

so_corpus = StackOverflowCorpus()
print(so_corpus)

# save the corpus
corpora.MmCorpus.serialize('so_bow_corpus.mm', so_corpus)
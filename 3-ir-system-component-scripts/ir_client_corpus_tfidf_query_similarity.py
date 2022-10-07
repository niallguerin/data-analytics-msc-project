from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.utils import simple_preprocess
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pandas as pd

# load dictionary and corpus objects
so_dictionary = corpora.Dictionary.load('so_dictionary.dict')
so_corpus = corpora.MmCorpus('so_bow_corpus.mm')
so_index = similarities.Similarity.load('so_tfidf_index.index')
# print(so_index)

# construct the model as we use it later for the index similarity test against the query vector
so_tfidf = models.TfidfModel(so_corpus, normalize=True)

# construct the query in keyword terms - our query test cases for the IR module and ML predictions
# Workflow:
# 1. These queries are fired against the IR module as single query test cases
# 2. The IR module will try to find the optimal resultset and pass the ML module layer
# 3. The ML Module will process the resultset and try to find the single best answer prediction
# 4. The system will provide a result snippet with weblink in CLI to the SO website with answer
# These are based on common past queries I had in work or in college projects
query1 = "XML Processing in Python"
query2 = "How do I calculate someone's age in C#"
query3 = "SQL stored procedure web service"
query4 = "preferred way to remove spaces from a string in C++"
query5 = "pandas count NaN"

query = query1
print("Input Question Query: ", query)
query_vec_bow = so_dictionary.doc2bow(query.lower().split())

# convert the query to TFIDF space and display it as numeric vector to console
query_vec_tfidf = so_tfidf[query_vec_bow]
print(query_vec_tfidf)

# query - corpus similarity measure
# api reference: https://radimrehurek.com/gensim/similarities/docsim.html
# https://radimrehurek.com/gensim/tutorial.html
sims = so_index[so_tfidf[query_vec_tfidf]]
# print(list(enumerate(sims)))
sims = sorted(enumerate(sims), key=lambda item: -item[1])
# print(sims)  # print sorted (document number, similarity score) 2-tuples

# print smaller range as console is missing initial values as can see they are truncated
# so only seeing lower results
precision_k15 = 15
precision_k10 = 10
precision_k5 = 5

# set filenames to match query test cases for precision in case we need to re-run repeat tests
# the purpose of this is to generate ML model prediction test cases so the downstream ML model
# pipeline can be trained:tested and optimised independent of the IR module activities thus
# facilitating a simulated resultset / workflow for the ML model activity and not having to re-run
# query inputs end-to-end every time.
filename_k15 = "resultlist_for_mapper_k15.csv"
filename_k10 = "resultlist_for_mapper_k10.csv"
filename_k5 = "resultlist_for_mapper_k5.csv"

# generate list of values and use to read into the mapper script for generating ml resultset
resultlist_for_mapper = []
for i in range(precision_k5):
	# append only the stack overflow post index not the ranking value
	resultlist_for_mapper.append(sims[i][0])
	print(sims[i])

df_resultlist_for_mapper = pd.DataFrame(resultlist_for_mapper, columns=['record_ids'])
df_resultlist_for_mapper.to_csv(filename_k5, index=False)

# results interpretation: document at integer value e.g. 38188 has similarity score of 48%.

# # load corpus text file
# corpus_library = pd.read_csv('data/corpus_so.txt', header=None)
# corpus_library.columns = ['posts']
from gensim import corpora
from gensim import similarities
from gensim.test.utils import get_tmpfile
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# this script does little other than load up the previously constructed dictionary and bag of words model for the
# active corpus being tested for query input string similarity in next script
# the key line is 16 where the Similarity matrix structure is split across shards to disc into a series of sub-files
so_dictionary = corpora.Dictionary.load('so_dictionary.dict')
so_corpus = corpora.MmCorpus('so_bow_corpus.mm')
# print(so_corpus)

# generate the tf-idf matrix index (built as shards across disk - shards stored in var/*, index stored here
index_tmpfile = get_tmpfile("index")
so_index = similarities.Similarity(index_tmpfile, so_corpus, num_features=len(so_dictionary))
# this file links to the index files so be careful over-writing it assuming it will work if you generate a new index
# for a new corpus file with same names as old shard indices will be over-written from what I observed on Windows; expected
# but something to be aware if errors when loading the indices later in the query similarity script.
# Also try and avoid crazy values in num_features attribute. Once it gets over 50K-100K (and mine did),
# gensim developers say to consider transforming to LSI models rather than tf-idf models
so_index.save("so_tfidf_index.index")
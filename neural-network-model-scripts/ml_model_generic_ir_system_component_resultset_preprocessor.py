# preprocessor to reconfigure the columns and then drop the columns on the IR system component resultset output csv
import pandas as pd

# !!!IMPORTANT IN THIS FILE SET THE INPUT FILE AND THE OUTPUT FILE VARIABLE NAME WHEN RUNNING THE TEST CASE!!!

# configure list of input files to allow repeat tests from only this point in workflow
infile_q1_k15 = "so_qs_resultset_answers00_q1_k15.csv"
infile_q1_k10 = "so_qs_resultset_answers00_q1_k10.csv"
infile_q1_k5 = "so_qs_resultset_answers00_q1_k5.csv"

infile_q2_k15 = "so_qs_resultset_answers00_q2_k15.csv"
infile_q2_k10 = "so_qs_resultset_answers00_q2_k10.csv"
infile_q2_k5 = "so_qs_resultset_answers00_q2_k5.csv"

infile_q3_k15 = "so_qs_resultset_answers00_q3_k15.csv"
infile_q3_k10 = "so_qs_resultset_answers00_q3_k10.csv"
infile_q3_k5 = "so_qs_resultset_answers00_q3_k5.csv"

infile_q4_k15 = "so_qs_resultset_answers00_q4_k15.csv"
infile_q4_k10 = "so_qs_resultset_answers00_q4_k10.csv"
infile_q4_k5 = "so_qs_resultset_answers00_q4_k5.csv"

infile_q5_k15 = "so_qs_resultset_answers00_q5_k15.csv"
infile_q5_k10 = "so_qs_resultset_answers00_q5_k10.csv"
infile_q5_k5 = "so_qs_resultset_answers00_q5_k5.csv"

# read the file into a dataframe: adjust the infile*q*k* variable to match query number and precision@ test case
# watch separator; needs to be , not ; on these input data files
df_train = pd.read_csv(infile_q5_k5, sep=",", engine='c', escapechar='\\')
# print(df_train.head())

df_train['solutionlabel'] = df_train['solution']
# print(df_train.head())

df_train = df_train[['question_id','answers_count','time_difference','time_difference_rank','len','len_rank','wordcount','wordcount_rank','avg_chars_per_word','avg_chars_per_word_rank','sentences','sentences_rank','avg_words_per_sentence','avg_words_per_sentence_rank','longest_sentence','longest_sentence_rank','loglikelihood','loglikelihood_ascending_rank','F-K','F-K_ascending_rank','upvotes','upvotes_rank','has_links','solutionlabel']]
# print(df_train.head())

# do conversion for column format of values
df_train['has_links'] = df_train['has_links'].map({True: 1, False: 0})
df_train['solutionlabel'] = df_train['solutionlabel'].map({True: 1, False: 0})
# print(df_train.head())

# rename column
df_train = df_train.rename(columns={'solutionlabel': 'solution'})
print(df_train.head())

# print(len(df_train.columns))

# print the file to csv

# configure the output file list so they have names matching query experiments to allow quick retesting of models
outfile_q1_k15 = "df_resultset_answers00_q1_k15.csv"
outfile_q1_k10 = "df_resultset_answers00_q1_k10.csv"
outfile_q1_k5 = "df_resultset_answers00_q1_k5.csv"

outfile_q2_k15 = "df_resultset_answers00_q2_k15.csv"
outfile_q2_k10 = "df_resultset_answers00_q2_k10.csv"
outfile_q2_k5 = "df_resultset_answers00_q2_k5.csv"

outfile_q3_k15 = "df_resultset_answers00_q3_k15.csv"
outfile_q3_k10 = "df_resultset_answers00_q3_k10.csv"
outfile_q3_k5 = "df_resultset_answers00_q3_k5.csv"

outfile_q4_k15 = "df_resultset_answers00_q4_k15.csv"
outfile_q4_k10 = "df_resultset_answers00_q4_k10.csv"
outfile_q4_k5 = "df_resultset_answers00_q4_k5.csv"

outfile_q5_k15 = "df_resultset_answers00_q5_k15.csv"
outfile_q5_k10 = "df_resultset_answers00_q5_k10.csv"
outfile_q5_k5 = "df_resultset_answers00_q5_k5.csv"

df_train.to_csv(outfile_q5_k5, encoding='utf-8', index=False, header=False)
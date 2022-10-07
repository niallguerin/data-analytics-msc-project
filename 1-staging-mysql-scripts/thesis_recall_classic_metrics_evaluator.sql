-- utility script to record recall metrics on information retrieval and filtering client result set versus corpus test case
-- https://stackoverflow.com/questions/3635166/how-to-import-csv-file-to-mysql-table
-- troubleshooting: https://stackoverflow.com/questions/12179770/mysql-1062-duplicate-entry-0-for-key-primary
use stackoverflow;
LOAD DATA INFILE  
'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/relevance_metrics/so_corpus_metrics_all_answers00.csv'
INTO TABLE metrics1  
-- field is terminated by new line, line is terminated by new line
fields terminated by '\n'
lines terminated by '\n'
(document);

-- for q5 test case
-- utility script to record recall metrics on information retrieval and filtering client result set versus corpus test case
-- https://stackoverflow.com/questions/3635166/how-to-import-csv-file-to-mysql-table
-- troubleshooting: https://stackoverflow.com/questions/12179770/mysql-1062-duplicate-entry-0-for-key-primary
use stackoverflow;
LOAD DATA INFILE  
'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/relevance_metrics/so_corpus_metrics_all_answers9080.csv'
INTO TABLE metrics3  
-- field is terminated by new line, line is terminated by new line
fields terminated by '\n'
lines terminated by '\n'
(document);
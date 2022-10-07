--Modified by Niall Guerin on 11.06.2019
--Utility sql script to export answers from MySQL database to CSV data format for machine learning pipeline
--This will need to be updated to reflect any underlying schema changes and dependent schema.sql files before we run ML pipeline out of box from Calefato

--https://stackoverflow.com/questions/32737478/how-should-i-tackle-secure-file-priv-in-mysql

select PQ.Id, PQ.Body, PQ.Title, PQ.Tags, PQ.AcceptedAnswerId, PQ.AnswerCount, PQ.CreationDate, PA.Id,
       PA.CreationDate, PA.Score, PA.Body
from Posts as PA, Posts as PQ   
where PA.PostTypeId = 2 and PA.ParentId = PQ.Id
order by PA.ParentId
-- On Windows, use forward-slash, not back-slash
-- Use SHOW VARIABLES LIKE "secure_file_priv"; to see the correct path location on file system
INTO OUTFILE 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/august_all_answers_validated_1.csv' 
fields terminated by ';' enclosed by '"';

-- re-run on August 15th 2019 and larger resultset for corpus obtained now matching almost full size of the db posts table
-- result: 27107580 row(s) affected
-- 19:36:53	select PQ.Id, PQ.Body, PQ.Title, PQ.Tags, PQ.AcceptedAnswerId, PQ.AnswerCount, PQ.CreationDate, PA.Id,        PA.CreationDate, PA.Score, PA.Body from Posts as PA, Posts as PQ    where PA.PostTypeId = 2 and PA.ParentId = PQ.Id order by PA.ParentId -- On Windows, use forward-slash, not back-slash -- Use SHOW VARIABLES LIKE "secure_file_priv"; to see the correct path location on file system INTO OUTFILE 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/august_all_answers_aug_validated_1.csv'  fields terminated by ';' enclosed by '"'	27107580 row(s) affected	54468.031 sec


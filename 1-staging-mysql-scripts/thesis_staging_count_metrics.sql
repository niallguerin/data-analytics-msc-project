use stackoverflow;

-- Fetch answer posts from Posts Table
-- select COUNT(Id) as NumberOfRecords
-- from Posts where PostTypeId = 2 and ParentId = Id

-- select COUNT(Id) as NumberOfRecords
-- from Posts where CreationDate >= '2018-01-01 01:00:00' and CreationDate < '2019-01-01 01:00:00'
-- result = 4673821

select COUNT(Id) as NumberOfRecords
from Posts where CreationDate >= '2017-01-01 01:00:00'
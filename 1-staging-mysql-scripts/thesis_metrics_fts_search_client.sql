use stackoverflow;

-- do full text search using one of the availabe FTS modes from boolean to natural language

-- natural language mode
SELECT id, document, MATCH document AGAINST ('XML Processing in Python' IN NATURAL LANGUAGE MODE) as relevance FROM metrics2 group by relevance order by count(relevance > 0), relevance desc;

-- boolean mode
SELECT id, document, MATCH (document) AGAINST ('XML Processing in Python' IN BOOLEAN MODE) as relevance FROM metrics2 group by relevance order by count(relevance > 0), relevance desc;
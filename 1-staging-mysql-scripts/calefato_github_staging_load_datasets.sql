use stackoverflow;
load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Badges.xml'
into table Badges
rows identified by '<row>';

 load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Comments.xml'
 into table Comments
 rows identified by '<row>';

 /* load xml infile '/path/to.../PostHistory.xml' */
 /* into table PostHistory */
 /* rows identified by '<row>'; */

 load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/PostLinks.xml'
 INTO TABLE PostLinks
 ROWS IDENTIFIED BY '<row>';

 load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Posts.xml'
 into table Posts
 rows identified by '<row>';

 load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Tags.xml'
 INTO TABLE Tags
 ROWS IDENTIFIED BY '<row>';

 load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Users.xml'
 into table Users
 rows identified by '<row>';

 load xml infile 'D:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Votes.xml'
 into table Votes
 rows identified by '<row>';

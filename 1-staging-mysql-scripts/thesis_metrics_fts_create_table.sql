use stackoverflow;

CREATE TABLE `metrics2` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `document` text,
  PRIMARY KEY (`id`),
  FULLTEXT (document)
) ENGINE=InnoDB AUTO_INCREMENT=000001 DEFAULT CHARSET=UTF8MB4

use stackoverflow;

CREATE TABLE `metrics3` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `document` text,
  PRIMARY KEY (`id`),
  FULLTEXT (document)
) ENGINE=InnoDB AUTO_INCREMENT=000001 DEFAULT CHARSET=UTF8MB4
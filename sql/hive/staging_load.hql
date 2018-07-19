DROP TABLE IF EXISTS EDW.EDIFICE;
CREATE EXTERNAL TABLE IF NOT EXISTS EDW.EDIFICE
(
	ID STRING,
	account	STRING,
	upc	STRING,
	storenumber	STRING,
	qs	INT,
	qa	INT,
	qr	INT,
	qu	INT,
	xr	DOUBLE,
	flag	CHAR(1),
	lastUPD	TIMESTAMP
)
PARTITIONED BY (Retailer STRING,year_day DATE)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION 'adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/staging';

INSERT OVERWRITE TABLE EDW.EDIFICE PARTITION(Retailer,year_day) 
SELECT CONCAT(cast(UPC as STRING), Retailer, CAST(weekending as STRING), CAST(Storenumber as STRING)) as ID,
account,
upc,
storenumber,
qs,
qa,
qr,
qu,
xr,
flag,
from_unixtime(unix_timestamp(CURRENT_TIMESTAMP)) AS lastUPD,
Retailer,
weekending AS year_day
FROM DEFAULT.EDIFICE_STG;


: set ff=unix
#!/bin/bash -e
hive2server=jdbc:hive2://headnodehost:10001/yetitest?hive.server2.transport.mode=http
blobpath=wasb://yetidatalake01@yetidatalake01.blob.core.windows.net/spark05/hive_backup/yetidm
mkdir -p /home/yeti/Spark05_archive/hive_backup/logs/import/yetidm
logpath=/home/yeti/Spark05_archive/hive_backup/logs/import/yetidm
mkdir -p /home/yeti/Spark05_archive/hive_backup/hivetables
filepath=/home/yeti/Spark05_archive/hive_backup/hivetables

<<COMMENT1
echo "Spark04:yetidm:finding tables on yetidm database `date` ";
hive -S -e "use yetidm;show tables;" > $filepath/yetidm
COMMENT1

while read line
do
i=`echo $line | awk '{print $1}'`
(
beeline -u $hive2server -e "create database if not exists yetidm_spark05;use yetidm_spark05; import table $i from '$blobpath/$i';" >$logpath/$i 2>&1 && 
	(cat $logpath/$i && echo -e "dl01-dev:hive table $i imported from $blobpath/$i") ||
	((cat $logpath/$i && echo -e "dl01-dev:$i:Failed to import Hive table $i from $blobpath/$i") | mail -s "dl01-dev:Failed to import Hive table $i" areddy@yeticoolers.com)
)
done<$filepath/yetidm

: set ff=unix
#!/bin/bash -e
hive2server=jdbc:hive2://headnodehost:10001/yetitest?hive.server2.transport.mode=http
blobpath=wasb://yetidatalake01@yetidatalake01.blob.core.windows.net/spark05/hive_backup/as_adventure
mkdir -p /home/yeti/Spark05_archive/hive_backup/logs/import/as_adventure
logpath=/home/yeti/Spark05_archive/hive_backup/logs/import/as_adventure
mkdir -p /home/yeti/Spark05_archive/hive_backup/hivetables
filepath=/home/yeti/Spark05_archive/hive_backup/hivetables

<<COMMENT1
echo "Spark04:Default:finding tables on default database `date` ";
hive -S -e "use default;show tables;" > $filepath/as_adventure
COMMENT1

while read line
do
i=`echo $line | awk '{print $1}'`
(
beeline -u $hive2server -e "create database if not exists as_adventure;use as_adventure; import table $i from '$blobpath/$i';" >$logpath/$i 2>&1 && 
	(cat $logpath/$i && echo -e "dl01-dev:hive table $i imported from $blobpath/$i") ||
	((cat $logpath/$i && echo -e "dl01-dev:$i:Failed to import Hive table $i from $blobpath/$i") | mail -s "dl01-dev:Failed to import Hive table $i" areddy@yeticoolers.com)
)
done<$filepath/as_adventure
: set ff=unix
#!/bin/bash -e
hive2server=jdbc:hive2://headnodehost:10001/default?hive.server2.transport.mode=http
hqlpath=/home/avinash_r/OSF/CRMREPORT
rawPath=adl://yetiadls.azuredatalakestore.net/clusters/data/01_raw/osf/crmreport
stagingPath=adl://yetiadls.azuredatalakestore.net/clusters/data/02_staging/osf/crmreport
archivePath=adl://yetiadls.azuredatalakestore.net/clusters/data/05_archive/osf/crmreport

hdfs dfs -mkdir -p $stagingPath/01_stg
hdfs dfs -mv $rawPath/* $stagingPath/01_stg/

#beeline -u $hive2server --hiveconf hive.query.name=CRMReport_staging -f $hqlpath/CRMReport_staging.hql

hdfs dfs -mkdir -p $archivePath/01_stg
hdfs dfs -mv $stagingPath/01_stg/* $archivePath/

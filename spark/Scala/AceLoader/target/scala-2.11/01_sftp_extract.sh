: set ff=unix
#!/bin/bash -e
adlspath=adl://yetiadls.azuredatalakestore.net/clusters/data/01_raw/osf
filepath=/home/xsbdsa/OSF
mkdir -p $filepath/osfcsv
osfpath=$filepath/osfcsv
mkdir -p $filepath/logfile
logfile=$filepath/logfile
mkdir -p $filepath/input
inputpath=$filepath/input
mkdir -p $filepath/output
outputpath=$filepath/output
loadDateTime=`date '+%Y%m%d %H:%M:%S'`

echo "DPE-3600:Start executing osf workflows at `date '+%Y%m%d %H:%M:%S'` ";

echo "Importing today's osf files from amazon sftp server"
sftp azurechsftp1@interface1.yeti.com <<EOF
lcd $osfpath
get /transfer/*
# rename transfer/ archive/`date '+%Y%m%d'`/
# mkdir transfer
bye
EOF

echo "remove space in the file names at `date '+%Y%m%d %H:%M:%S'`"
rename 's/ //g' $osfpath/*

echo "upload files to ADLS at `date '+%Y%m%d %H:%M:%S'`"
for i in $osfpath/*.csv; do
(
reportName="$(echo "${i,,}" | cut -d "-" -f1| cut -d "/" -f6)"
fileName="$(echo "$i" | cut -d "/" -f6)"
hadoop fs -mkdir -p "$adlspath/$reportName"
hadoop fs -rm "$adlspath/$reportName/$fileName"
hadoop fs -put $i "$adlspath/$reportName/$fileName"
# mkdir -p "$outputpath/$loadDateTime"
# mv "$i" "$outputpath/$loadDateTime/$i"
)
done

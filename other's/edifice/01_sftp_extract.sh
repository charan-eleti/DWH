: set ff=unix
#!/bin/bash -e
adlspath=adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/input
filepath=/home/xsbdsa/EDIFICE/cron
mkdir -p $filepath/edificecsv
edificepath=$filepath/edificecsv
mkdir -p $filepath/input
inputpath=$filepath/input
mkdir -p $filepath/output
outputpath=$filepath/output
loadDateTime=`date '+%Y%m%d %H:%M:%S'`

echo "DPE-3600:Start executing Edifice workflows at `date '+%Y%m%d %H:%M:%S'` ";

echo "Importing current weeks edifice files from sftp server"
sshpass -p 'mechu8aW' sftp -oHostKeyAlgorithms=+ssh-dss -o KexAlgorithms=diffie-hellman-group14-sha1 YETICOOLERS@carsftp.edificeinfo.com <<EOF
lcd $edificepath
get sftproot:/OUTBOX/BACKUP/20180430/*
bye
EOF

echo "remove space in the file names at `date '+%Y%m%d %H:%M:%S'`"
rename 's/ //g' $edificepath/*

echo "upload files to ADLS at `date '+%Y%m%d %H:%M:%S'`"
for i in $edificepath/*.txt; do
(
filename="$(echo "$i" | cut -d "_" -f1 | cut -d "/" -f7)""_""$(echo "$i" | cut -d "_" -f2)""_""$(echo "$i" | cut -d "_" -f3)""_""$(echo "$i" | cut -d "_" -f4)"
hadoop fs -mkdir -p "$adlspath"
hadoop fs -put $i "$adlspath/$filename"
mkdir -p "$outputpath/$loadDateTime"
mv "$i" "$outputpath/$loadDateTime/$filename"
)
done


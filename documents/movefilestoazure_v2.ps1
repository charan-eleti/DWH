param([string]$BasePath = "", 
[string]$Database = "E10DBTEST", 
[string]$Server = "WS08R2SQL02.AD.yeticoolers.com",
[string]$subscriptionID = "b7ca7484-aaf7-4da5-ae41-28e3021f41cf",
[string]$httpUserName = "admin",  #HDInsight cluster username
[string]$httpPassword = 'aKexT0ZG53$$TyPebVc',#call Kevin to get password
#[string]$httpUserName = "kjahn@yeticoolers01.onmicrosoft.com",  #HDInsight cluster username
#[string]$httpPassword = 'xxxxxxxxxxxxxxxx',
[string]$hdinsightClusterName = 'yetispark04',
[string]$defaultStorageAccountName = 'yetidatalake01',
[string]$defaultStorageAccountKey = 'w7ysCwHk03SjVDJilaQktV+I1Jnx/+OtbqMHzERwPV/xyGdkJYGtmnUh6cSwuS+rFQlEt+0izjIZ4FuepxOhyg==',
[string]$resourceGroupName = 'yetidatamart01',
[string]$fileloc = 'C:\IRAC\Spark05\archive\flume',
[string]$destpath = 'C:\IRAC\Spark05\archive\flume_archive'
)

#for recursive shorter method run this
#cd [root-data-folder]
#ls –Recurse –Path $localFolder | Set-AzureStorageBlobContent –Container $containerName –Context $blobContext


# HDInsight variables
$defaultBlobContainerName = $defaultStorageAccountName


clear

$DATETIMENOW = date

#region - Connect to Azure subscription
Write-Host "`nConnecting to your Azure subscription ..." -ForegroundColor Green
try{Get-AzureRmContext}
catch{Login-AzureRmAccount}
#endregion


# Validate the cluster
Write-Host "`nValidating HDInsight Cluster ..." -ForegroundColor Green
Get-AzureRmHDInsightCluster -ClusterName $hdinsightClusterName
#endregion

# Define the connection string
$storageConnectionString = "DefaultEndpointsProtocol=https;AccountName=$defaultStorageAccountName;AccountKey=$defaultStorageAccountKey"

# Azure subscription-specific variables.
$storageAccountName = $defaultStorageAccountName
$containerName = $defaultBlobContainerName

# Find the local folder where this PowerShell script is stored.
#$currentLocation = Get-location
#$thisfolder = Split –parent $currentLocation

# Upload files in data subfolder to Azure.
$localfolder = $fileloc
$destfolder = "/spark05/archive"
$storageAccountKey = $defaultStorageAccountKey
$blobContext = New-AzureStorageContext -StorageAccountName $storageAccountName -StorageAccountKey $storageAccountKey



# loop for each file and upload to blob storage
$files = Get-ChildItem $localFolder -Filter $pattern | ?{ $_.fullname -notmatch "\\Archive\\?" -and $_.fullname -notmatch "\\temp\\?" } |  where {$_.Attributes -ne "Directory" }
foreach($file in $files)
{
  write $file.name

  #Do this if you want to place the file in a folder and change blobname to destfolder2
  $pos = $file.name.IndexOf(".")
  $destfolder2 = $destfolder+'\'+$file.name.Substring(0, $pos)


  $fileName = "$localFolder\$file"
  $blobName = "$destfolder/$file"
  write-host "copying $fileName to $blobName"
  Set-AzureStorageBlobContent -File $filename -Container $containerName -Blob $blobName -Context $blobContext -Force
  Move-ITem $fileName $DestPath\$file -force
  $pos = 0
  $foldercnt = 0
  $foldercnt2 = 0
} 
write-host "All files in $localFolder uploaded to $containerName!"


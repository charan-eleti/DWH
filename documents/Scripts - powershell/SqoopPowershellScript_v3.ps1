param([string]$BasePath = "", 
[string]$Database = "yetiepicorstg", 
[string]$Server = "YETILP-336\SQLEXPRESS:1433",
[string]$subscriptionID = "b7ca7484-aaf7-4da5-ae41-28e3021f41cf",
[string]$httpUserName = "admin",  #HDInsight cluster username
[string]$httpPassword = 'aKexT0ZG53$$TyPebVc',#call Kevin to get password
#[string]$httpUserName = "kjahn@yeticoolers01.onmicrosoft.com",  #HDInsight cluster username
#[string]$httpPassword = 'xxxxxxxxxx',
[string]$hdinsightClusterName = 'yetispark04',
[string]$defaultStorageAccountName = 'yetidatalake01',
[string]$defaultStorageAccountKey = 'w7ysCwHk03SjVDJilaQktV+I1Jnx/+OtbqMHzERwPV/xyGdkJYGtmnUh6cSwuS+rFQlEt+0izjIZ4FuepxOhyg==',
[string]$resourceGroupName = 'yetidatamart01'
)
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

write "xyz"
# Define the connection string
$storageConnectionString = "DefaultEndpointsProtocol=https;AccountName=$defaultStorageAccountName;AccountKey=$defaultStorageAccountKey"

# jdbc connection
$JDBCconnectionString = "jdbc:sqlserver://$SERVER;databaseName=$Database;"

# table name to transfer
$hivetableName = "w3c"
# hive database
$hivedir = "hive\warehouse\w3c"
$securepw = $httpPassword|ConvertTo-SecureString  -AsPlainText -Force
#$httpPassword = ConvertTo-SecureString -String 'aKexT0ZG53$$TyPebVc' -AsPlainText -Force $httpCredential = New-Object -TypeName System.Management.Automation.PSCredential ArgumentList $httpUserName, $httpPassword
$httpCredential = New-Object System.Management.Automation.PSCredential("$httpUserName",$securepw)

#$querystring = "SELECT *, GETDATE() AS DwLastUpdated FROM w3c"

$sqoopDef = New-AzureRmHDInsightSqoopJobDefinition `
    -Command "import --connect 'jdbc:sqlserver://YETILP-336\SQLEXPRESS:1433;databaseName=yetiepicorstg' --username avinash --password areddy --table $hivetablename  --target-dir $hivedir --fields-terminated-by '\t' --lines-terminated-by '\n' -m 1"
 #   -Command "import --connect $JDBCconnectionString --query $querystring --split-by EL.Name --hive-import --hive-table $hivetablename --target-dir $hivedir"--driver com.microsoft.sqlserver.jdbc.SQLServerDriver

$sqoopJob = Start-AzureRmHDInsightJob `
                -ClusterName $hdinsightClusterName `
                -HttpCredential $httpCredential `
                -JobDefinition $sqoopDef #-Debug -Verbose

Wait-AzureRmHDInsightJob `
    -ResourceGroupName $resourceGroupName `
    -ClusterName $hdinsightClusterName `
    -HttpCredential $httpCredential `
    -JobId $sqoopJob.JobId

Write-Host "Standard Error" -BackgroundColor Green
Get-AzureRmHDInsightJobOutput `
    -ResourceGroupName $resourceGroupName `
    -ClusterName $hdinsightClusterName `
    -DefaultStorageAccountName $defaultStorageAccountName `
    -DefaultStorageAccountKey $defaultStorageAccountKey `
    -DefaultContainer $defaultBlobContainerName `
    -HttpCredential $httpCredential `
    -JobId $sqoopJob.JobId `
    -DisplayOutputType StandardError

Write-Host "Standard Output" -BackgroundColor Green
Get-AzureRmHDInsightJobOutput `
    -ResourceGroupName $resourceGroupName `
    -ClusterName $hdinsightClusterName `
    -DefaultStorageAccountName $defaultStorageAccountName `
    -DefaultStorageAccountKey $defaultStorageAccountKey `
    -DefaultContainer $defaultBlobContainerName `
    -HttpCredential $httpCredential `
    -JobId $sqoopJob.JobId `
    -DisplayOutputType StandardOutput


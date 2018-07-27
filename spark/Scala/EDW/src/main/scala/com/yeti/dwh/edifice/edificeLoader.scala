package com.yeti.dwh.edifice

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SaveMode, SparkSession}

object edificeLoader{
  /*
    //Usage:
      spark-submit --class com.yeti.dwh.edifice.edificeLoader \
      --master yarn \
      edw_2.11-1.1.6.jar \
      adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/input \
      adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/processed \
      adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/target \
      adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/archive
*/
  def main(args: Array[String]) {
    val inputPath = args(0)
    val processedPath = args(1)
    val hiveTablePath = args(2)
    val backupPath = args(3)
    //val loadDateTime = new SimpleDateFormat("yyyy-MM-dd HH:MM:SS").format(Calendar.getInstance().getTime)
    val spark = SparkSession.builder
      //.master("local")
      .appName("edificeLoder")
      //.config("spark.sql.warehouse.dir", "file:///C:/temp")
      .getOrCreate()
    val sc = spark.sparkContext
    //val rootLogger = Logger.getRootLogger()
    //rootLogger.setLevel(Level.WARN)
    //val dataRecRDD = sc.wholeTextFiles(inputPath + "/*") //get files from input folder
    val dataRecRDD = sc.wholeTextFiles(inputPath + "/*") //get files from input folder
    val data_rec = dataRecRDD.flatMap(x => edificeReport.parse(x))
    val dataRecDF = spark.createDataFrame(data_rec,edificeReport.schema)
    val dataRecLastUPDDF = dataRecDF.withColumn("lastUPD", from_unixtime(unix_timestamp(current_timestamp)))
    dataRecLastUPDDF.printSchema()
    //dataRecLastUPDDF.show()
    //dataRecDF.write.partitionBy("retailer","year_day").format("csv").mode(SaveMode.Overwrite).option("retailer1", "retailer").save(args(1) + "/" + data_rec(0) + ".csv")
    /*
        dataRecDF.write.partitionBy("retailer","year_day")
          .format("com.databricks.spark.csv").mode(SaveMode.Overwrite)
          .option("header", "true")
          .save(args(1)) //save to processed folder
    */
    dataRecLastUPDDF.write.partitionBy("retailer","year_day")
      .option("header", "false")
      .mode(SaveMode.Overwrite)
      //.option("delimiter", ",")
      .csv(processedPath)//save to processed folder
    println(">>> Edifice input files in " + inputPath + " have been processed and copied to folder " + processedPath)
    println(">>>EdificeLoader Ends here")

    println("<<<delete _SUCCESS file in processed folder " + processedPath)
    if(HDFSUtil.exists(processedPath + "/" + "_SUCCESS")) {
      HDFSUtil.deleteDirectory(processedPath + "/" + "_SUCCESS") // delete _SUCCESS file in processed folder
    }
    println("<<<move child directories from " + processedPath + " to " +  hiveTablePath)
    HDFSUtil.moveNestedDir(processedPath, hiveTablePath) //move child directories from processed folder to target directory
    //fsUtil.moveFileOrDir(inputPath, backupPath + "/loadDateTime=" + loadDateTime) //archive input files after processing for backup
    println("<<<archive input files from " + inputPath + " to " +  backupPath)
    HDFSUtil.moveInputFiles(inputPath, backupPath) //archive input files after processing for backup
    println("<<<create inputFiles Directory " + inputPath)
    HDFSUtil.createDirectory(inputPath) //create inputFiles Directory for delta loads
    println(">>>HDFSUtil Ends here")

  }
}


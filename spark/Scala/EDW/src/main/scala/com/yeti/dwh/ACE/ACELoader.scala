package com.yeti.dwh.ACE

import org.apache.log4j._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, SparkSession}

object ACELoader {
  val struct = StructType(
    StructField("Site", StringType, nullable = true) ::
      StructField("SiteDescription", StringType, nullable = true) ::
      StructField("Customer_#", StringType, nullable = true) ::
      StructField("Customer_Name", StringType, nullable = true) ::
      StructField("Address_Line_1", StringType, nullable = true) ::
      StructField("Address_Line_2", StringType, nullable = true) ::
      StructField("City", StringType, nullable = true) ::
      StructField("State", StringType, nullable = true) ::
      StructField("ZipCode", StringType, nullable = true) ::
      StructField("Business_Class", StringType, nullable = true) ::
      StructField("Format", StringType, nullable = true) ::
      StructField("Manufacturer_#", StringType, nullable = true) ::
      StructField("Manufacturer_Class_Code", StringType, nullable = true) ::
      StructField("Merchandise_Class", StringType, nullable = true) ::
      StructField("Product_Group", StringType, nullable = true) ::
      StructField("Article", StringType, nullable = true) ::
      StructField("Article_Name", StringType, nullable = true) ::
      StructField("Eaches_Item", StringType, nullable = true) ::
      StructField("Eaches_Cost", StringType, nullable = true) :: Nil
  )
  def main(args: Array[String]) {
    /*
  //Usage:
    spark-submit --class com.yeti.dwh.edifice.edificeLoader \
    --master yarn \
    --deploy-mode cluster \
    com.yeti.edw.ACE.jar \
    adl://yetiadls.azuredatalakestore.net/clusters/raw/custom_data_uploads/sales/ace \
    adl://yetiadls.azuredatalakestore.net/clusters/data/02_staging/ace \
    adl://yetiadls.azuredatalakestore.net/clusters/data/03_transformed/ace \
    adl://yetiadls.azuredatalakestore.net/clusters/data/05_archive/ace
    */
    val inputPath = args(0)
    //val inputPath = "adl://yetiadls.azuredatalakestore.net/clusters/raw/custom_data_uploads/sales/ace"
    val processedPath = args(1)
    val hiveTablePath = args(2)
    val backupPath = args(3)
    Logger.getLogger("org").setLevel(Level.ERROR)
    val fileName = HDFSUtil.listDirectories(inputPath, false)
    val year_day = fileName(0).splitAt(9)._2.splitAt(10)._1
    //val loadDateTime = new SimpleDateFormat("yyyy-MM-dd HH:MM:SS").format(Calendar.getInstance().getTime)
    val spark = SparkSession.builder
      //.master("local")
      .appName("ACELoader")
      .getOrCreate()
    val inputDF = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .option("delimiter", ",")
      .option("quote", "\"")
      .schema(struct)
      .load(inputPath)
    //val indexed = inputDF.withColumn("index", monotonically_increasing_id())
    //val filtered = indexed.filter(col("index") > 1).drop("index")
    val outputDF = inputDF.withColumn("year_day", lit(year_day))
    outputDF.printSchema()
    //outputDF.show(5)
    outputDF.write.partitionBy("year_day")
      .option("header", "false")
      .mode(SaveMode.Overwrite)
      //.option("delimiter", ",")
      .csv(processedPath) //save to processed folder
    println(">>> ACE input files in " + inputPath + " have been processed and copied to folder " + processedPath)
    println("<<<delete _SUCCESS file in processed folder " + processedPath)
    if (HDFSUtil.exists(processedPath + "/" + "_SUCCESS")) {
      HDFSUtil.deleteDirectory(processedPath + "/" + "_SUCCESS") // delete _SUCCESS file in processed folder
    }
    println("<<<move child directories from " + processedPath + " to " + hiveTablePath)
    HDFSUtil.moveNestedDir(processedPath, hiveTablePath) //move child directories from processed folder to target directory
    //fsUtil.moveFileOrDir(inputPath, backupPath + "/loadDateTime=" + loadDateTime) //archive input files after processing for backup
    println("<<<archive input files from " + inputPath + " to " + backupPath)
    HDFSUtil.moveInputFiles(inputPath, backupPath) //archive input files after processing for backup
    println("<<<create inputFiles Directory " + inputPath)
    HDFSUtil.createDirectory(inputPath) //create inputFiles Directory for delta loads
    println(">>>HDFSUtil Ends here")
  }
}

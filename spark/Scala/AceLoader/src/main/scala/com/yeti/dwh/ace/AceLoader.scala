package com.yeti.dwh.ace

import org.apache.commons.io.FilenameUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, SparkSession}

object AceLoader {
  @transient lazy val LOG: Logger = Logger.getLogger(getClass.getName)

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

  def getFilesOfType(extension: String, path: String): List[String] = {
    val filePath = new Path(path)
    filePath
      .getFileSystem(new Configuration())
      .listStatus(filePath) // get directory listing
      .map(p => FilenameUtils.getName(p.getPath.toString)) // exract filename
      .filter(filename => extension match {
      case "*" => true
      case _ => FilenameUtils.getExtension(filename).toUpperCase.matches(extension)
    }).toList
  }

  def main(args: Array[String]) {
    val inputPath = args(0)
    //val inputPath = "adl://yetiadls.azuredatalakestore.net/clusters/data/01_raw/custom_data_uploads/sales/ace/"
    val processedPath = args(1)
    val hiveTablePath = args(2)
    val backupPath = args(3)
    Logger.getLogger("org").setLevel(Level.ERROR)
    val inputFiles = getFilesOfType("CSV", inputPath)
    try {
      // Return if none found
      if (inputFiles.isEmpty) {
        LOG.warn("Number of input files is zero, exiting!!")
        return
      }
      val dirs = HDFSUtil.listDirectories(inputPath, fullPath = false)
      //val loadDateTime = new SimpleDateFormat("yyyy-MM-dd HH:MM:SS").format(Calendar.getInstance().getTime)
      val spark = SparkSession.builder
        .master("yarn")
        .appName("ACELoader")
        .getOrCreate() //.set("spark.hadoop.validateOutputSpecs", "false")
      for (dirName <- dirs) {
        val year_day = dirName.splitAt(9)._2.splitAt(10)._1
        println("Year Day:" + year_day)
        val inputDF = spark.read.format("com.databricks.spark.csv")
          .option("header", "true")
          .option("inferSchema", "false")
          .option("delimiter", ",")
          .option("quote", "\"")
          .schema(struct)
          .load(inputPath + "/" + dirName)
        val outputDF = inputDF.withColumn("year_day", lit(year_day))
        outputDF.printSchema()
        outputDF.write //.partitionBy("year_day")
          .option("header", "false")
          .mode(SaveMode.Overwrite)
          //.option("delimiter", ",")
          .csv(processedPath + "/" + "year_day=" + year_day) //save to processed folder
        if (HDFSUtil.exists(processedPath + "/" + "year_day=" + year_day + "/" + "_SUCCESS")) {
          HDFSUtil.deleteDirectory(processedPath + "/" + "year_day=" + year_day + "/" + "_SUCCESS") // delete _SUCCESS file in processed folder
        }
      }
      println("<<<move child directories from " + processedPath + " to " + hiveTablePath)
      HDFSUtil.moveAceNestedDir(processedPath, hiveTablePath) //move child directories from processed folder to target directory
      //fsUtil.moveFileOrDir(inputPath, backupPath + "/loadDateTime=" + loadDateTime) //archive input files after processing for backup
      println("<<<archive input files from " + inputPath + " to " + backupPath)
      HDFSUtil.moveInputFiles(inputPath, backupPath) //archive input files after processing for backup
      println("<<<create inputFiles Directory " + inputPath)
      HDFSUtil.createDirectory(inputPath) //create inputFiles Directory for delta loads
      println(">>>HDFSUtil Ends here")
    }
    catch {
      case e: Exception =>
        LOG.warn(s"Ace Load Failure, exception occured ", e)
        throw e
    }
  }
}

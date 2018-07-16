package com.yeti.dwh.edifice

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SaveMode, SparkSession}

import scala.collection.mutable.ArrayBuffer



object EdificeLoader{
  // outputSchema for edifice table
  val schema = StructType(Array(
    StructField("ID", StringType),
    StructField("retailer", StringType),
    StructField("account", StringType),
    StructField("year_day", StringType),
    StructField("UPC", StringType),
    StructField("STORENUMBER", StringType),
    StructField("QS", StringType),
    StructField("QA", StringType),
    StructField("QR", StringType),
    StructField("QU", StringType),
    StructField("XR", StringType),
    StructField("flag", StringType),
    StructField("loadDate", StringType)
  ))

  //reportDate method will return date in specified format.
  def reportDate(s: String): String ={
    var  simpleDateFormat:SimpleDateFormat = new SimpleDateFormat("mm/dd/yyyy")
    var  date:Date = simpleDateFormat.parse(s)
    val year_day = new SimpleDateFormat("yyyymmdd").format(date)
    year_day
  }

  /* mapSections method returns a MAP[String, ArrayBuffer[String]]
      key : absolute File Path
      value : file content after splitting with keyword HDR
  */
  def mapSections(reportText: Array[String]) = {
    var reportBuffer: ArrayBuffer[String] = new ArrayBuffer
    var reportMap: Map[String, ArrayBuffer[String]] = Map()
    for (line <- reportText){
      val splitLine = line.split("\r\n")
      val header = splitLine.head
      // get retailer, Account, Year_day and Flag from header
      val splitHeader = header.split('|')
      val retailer = splitHeader(1).replace(" ", "_")
      val account = splitHeader(2).replace(" - ", "_").replace(" ", "_")
      val year_day = reportDate(splitHeader(3))
      val flag = splitHeader(4)
      //remove header and footer from split file
      val withoutHeaderFooter = splitLine.filter(line => line != header).dropRight(1)
      // append retailer, Account, Year_day and Flag for each account
      val accountLine = withoutHeaderFooter.map(x => retailer.trim + '|' + account.trim + '|' + year_day.trim + '|' + x + '|' + flag.trim)
      // file Name as Key in ReportMap
      val reportName = retailer + "_" + account + "_" + year_day
      for (line <- accountLine){
        reportBuffer += line
      }
      reportMap += (reportName -> reportBuffer)
      reportBuffer = new ArrayBuffer
    }
    reportMap
  }

  //This method splits file content for each account with key word "\r\nHDR"
  def splitSectionAndClean(text: String): Array[String] = {
    val splitLines = text.replace("\r\nHDR", "####HDR").split("####")
    splitLines
  }

  // This method converts ReportMap values into spark.sql.Row
  def parseData(txt: String): Row ={
    val txtSplitter = txt.split('|')
    val retailer = txtSplitter(0).trim
    val account = txtSplitter(1).trim
    val year_day = txtSplitter(2).trim
    val UPC = txtSplitter(3).trim
    val STORENUMBER = txtSplitter(4).trim
    val QS = txtSplitter(5).trim
    val QA = txtSplitter(6).trim
    val QR = txtSplitter(7).trim
    val QU = txtSplitter(8).trim
    val XR = txtSplitter(9).trim
    val flag = txtSplitter(10).trim
    val ID = UPC + retailer + year_day + STORENUMBER
    val loadDate = new SimpleDateFormat("yyyymmdd").format(Calendar.getInstance().getTime())
    Row(ID, retailer,account,year_day,UPC,STORENUMBER,QS,QA,QR,QU,XR,flag,loadDate)
  }

  def parse(keyValue: (String, String)) = {
    val reportMap = mapSections(splitSectionAndClean(keyValue._2))
    val r = new ArrayBuffer[Row]
    for ( line <- reportMap.values){
      for(reportLine <- line) {
        r += parseData(reportLine)
      }
    }
    r
  }

  def main(args: Array[String]) {

    //Usage: spark-submit edw_2.11-1.1.6.jar
    // adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/staging/input/*
    // adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/staging/output

    val spark = SparkSession.builder.
      master("local")
      .appName("edifice splitFiles")
      .config("spark.sql.warehouse.dir", "file:///C:/temp")
      .getOrCreate()
    val sc = spark.sparkContext
    val dataRecRDD = sc.wholeTextFiles("""C:\Intellij\EDW\src\main\resources\YETICOOLERS_ACADEMYSPORTS_ACADEMYSPORTS_20180624.txt""")
    val data_rec = dataRecRDD.flatMap(x => parse(x))
    val dataRecDF = spark.createDataFrame(data_rec,edificeReport.schema)
    dataRecDF.printSchema()
    dataRecDF.show()
    //dataRecDF.write.partitionBy("retailer","year_day").format("csv").mode(SaveMode.Overwrite).option("retailer1", "retailer").save(args(1) + "/" + data_rec(0) + ".csv")
    dataRecDF.write.partitionBy("retailer","year_day")
      .format("com.databricks.spark.csv").mode(SaveMode.Overwrite)
      .option("header", "true")
      .save(args(1))
    //HDFSUtil.deleteDirectory(args(2) + "/" + retailer + "/" + year_day)

  }
}


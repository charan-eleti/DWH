package com.yeti.dwh.ACE

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer

object ACEReport {
  // outputSchema for edifice table
  val schema = StructType(
      StructField("Site", StringType, nullable = true) ::
      StructField("Site Description", StringType, nullable = true) ::
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

  //reportDate method will return date in specified format.
  def reportDate(s: String): String = {
    var simpleDateFormat: SimpleDateFormat = new SimpleDateFormat("mm/dd/yyyy")
    var date: Date = simpleDateFormat.parse(s)
    val year_day = new SimpleDateFormat("yyyy-mm-dd").format(date)
    year_day
  }

  /* mapSections method returns a MAP[String, ArrayBuffer[String]]
      key : absolute File Path
      value : file content after splitting with keyword HDR
  */
  def mapSections(reportText: String) = {
    var reportBuffer: ArrayBuffer[String] = new ArrayBuffer
    var reportMap: Map[String, ArrayBuffer[String]] = Map()
    val reportLine = reportText.split("\r\n")
    val header = reportLine.head
    val ACELine = reportLine.filter(row => row != header)
    ACELine
  }

  //This method splits file content for each account with key word "\r\nHDR"
  def splitSectionAndClean(text : String): Array[String] = {
    val splitLines = text.replace("\r\n", "####").split("####")
    splitLines
  }

  // This method converts ReportMap values into spark.sql.Row
  def parseData(txt: String): Row = {
    val txtSplitter = txt.split(',')
    val Site = txtSplitter(0).trim
    val SiteDescription = txtSplitter(1).trim
    val Customer_# = txtSplitter(2).trim
    val Customer_Name = txtSplitter(3).trim
    val Address_Line_1 = txtSplitter(4).trim
    val Address_Line_2 = txtSplitter(5).trim
    val City = txtSplitter(6).trim
    val State = txtSplitter(7).trim
    val ZipCode = txtSplitter(8).trim
    val Business_Class = txtSplitter(9).trim
    val Format = txtSplitter(10).trim
    val Manufacturer_# = txtSplitter(11).trim
    val Manufacturer_Class_Code = txtSplitter(12).trim
    val Merchandise_Class = txtSplitter(13).trim
    val Product_Group = txtSplitter(14).trim
    val Article = txtSplitter(15).trim
    val Article_Name = txtSplitter(16).trim
    val Eaches_Item = txtSplitter(17).trim
    val Eaches_Cost = txtSplitter(18).trim
    val year_day = txtSplitter(19).trim
    //val lastUPD = new SimpleDateFormat("yyyy-MM-dd HH:MM:SS.SSSSSS").format(Calendar.getInstance().getTime)
    //val lastUPD = from_unixtime(unix_timestamp(current_timestamp())).toString()
    Row(Site, SiteDescription, Customer_#,
      Customer_Name, Address_Line_1, Address_Line_2,
      City, State, ZipCode, Business_Class,
      Format, Manufacturer_#, Manufacturer_Class_Code,
      Merchandise_Class, Product_Group, Article,
      Article_Name, Eaches_Item, Eaches_Cost,
      year_day)
    //Row(retailer, account, year_day, UPC, STORENUMBER, QS, QA, QR, QU, XR, flag, lastUPD)
  }

  def parse(keyValue: (String, String)) = {
    val ACELine = mapSections(keyValue._2)
    val fileName = keyValue._1
    val fileNameSplitter = fileName.split('_')
    val year_day = fileNameSplitter(2).trim
    val r = new ArrayBuffer[Row]
    for (line <- ACELine) {
        r += parseData(line + "," + year_day)
    }
    r
  }
}
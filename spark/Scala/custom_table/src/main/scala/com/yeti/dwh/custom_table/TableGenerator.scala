package com.yeti.dwh.custom_table

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._


object TableGenerator {def main(args: Array[String]): Unit = {
  val inputPath=args(0)
  val tableName=args(1)
  val inputUser=args(2)
    //val inputPath = "adl://yetiadls.azuredatalakestore.net/clusters/yeti-dpe-3600/custom_data_uploads/sales/Product_Categoy.csv"
    //val tableName = "edw_sales.product_catogory"
    //val inputUser = "nick"

  Logger.getLogger("org").setLevel(Level.ERROR)


  val spark = SparkSession
    .builder()
    .appName("TableGenerator")
    //.master("local[*]")
    //.config("spark.sql.warehouse.dir", "file:///C:/temp")
    .enableHiveSupport()
    .getOrCreate()

  val rawfile = spark.read.option("header","true").option("delimiter", "|").csv(inputPath)

  val finalDF = rawfile.withColumn("load_date_time",current_timestamp()).withColumn("inputed_by",lit(inputUser))

  import spark.implicits._

  //val schemaFinaldf=finalDF.toDS

   // finalDF.createOrReplaceTempView("source_table")

  val dropTb= spark.sql(s"DROP TABLE IF EXISTS $tableName")

  finalDF.write.mode(SaveMode.Overwrite).saveAsTable(tableName)

  //val sqlTest= spark.sql(s"Create Table $tableName as (Select *,Cast('$inputUser' as string) as User_name from source_table)")

  //val sqlTest= spark.sql("Select * from source_table limit 10").show

  spark.stop()

  }
}
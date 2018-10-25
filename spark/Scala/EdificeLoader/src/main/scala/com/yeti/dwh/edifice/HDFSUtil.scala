package com.yeti.dwh.edifice

import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

object HDFSUtil {
  private val conf = new Configuration()
  private val hdfsCoreSitePath = new Path("/etc/hadoop/conf/core-ste.xml")
  private val hdfsHDFSSitePath = new Path("/etc/hadoop/conf/hdfs-site.xml")
  conf.addResource(hdfsCoreSitePath)
  conf.addResource(hdfsHDFSSitePath)
  var fileSystem: FileSystem = _
  val fs: FileSystem = FileSystem.get(new URI("adl://yetiadls.azuredatalakestore.net"), conf)
  init(fs) //initialize hadoop fileSystem
  //init()

  def init(fs: FileSystem = null): Unit = {
    if(null == fs){
      fileSystem = FileSystem.get(conf)
    } else{
      fileSystem = fs
    }
  }

  //check if the directory exists
  def exists(path: String): Boolean = {
    val p = new Path(path)
    fileSystem.exists(p)
  }

  //drop and recreate the directory
  def recreateDirectory(path: String): Unit ={
    deleteDirectory(path)
    createDirectory(path)
  }

  //delete directory
  def deleteDirectory(path: String): Unit ={
    val dirPath = new Path(path)
    fileSystem.delete(dirPath, true)
  }

  //create directory
  def createDirectory(dirPath: String): Unit ={
    val path = new Path(dirPath)
    if(!fileSystem.exists(path)){
      fileSystem.mkdirs(path)
    }
  }

  //list child directories with or without absolute path.
  def listDirectories(path: String, fullPath: Boolean): List[String] = {
    var list: List[String] = List[String]()
    val status = fileSystem.listStatus(new Path(path))
    if(fullPath){
      status.foreach(x => list ::= x.getPath.toString)
    } else {
      status.foreach(x => list ::= x.getPath.getName)
    }
    list
  }

  //move files from source to target directory
  def moveFileOrDir(src: String, target: String): Unit = {
    val srcPath = new Path(src)
    val targetPath = new Path(target)
    if(fileSystem.exists(srcPath)){
      fileSystem.rename(srcPath, targetPath)
    }
  }

  //move child directories into target directory
  def moveNestedDir(srcDir: String, targetDir: String): Unit = {
    val srcFiles = listDirectories(srcDir, fullPath = false)
    for(srcFile <- srcFiles){
      if(exists(srcDir + "/" + srcFile)) {
        val targetFiles = listDirectories(srcDir + "/" + srcFile, fullPath = false)
        for (targetFile <- targetFiles) {
          if (exists(targetDir + "/" + srcFile + "/" + targetFile)) {
            deleteDirectory(targetDir + "/" + srcFile + "/" + targetFile)
            moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
          } else {
            if(!exists(targetDir + "/" + srcFile)) {
              createDirectory(targetDir + "/" + srcFile)
            }
            moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
          }
        }
      }
    }
  }

  //move inputFiles into archive directory
  def moveInputFiles(srcDir: String, targetDir: String): Unit = {
    val srcFiles = listDirectories(srcDir, fullPath = false)
    for(srcFile <- srcFiles){
      if(exists(srcDir + "/" + srcFile)) {
        if (exists(targetDir + "/" + srcFile)) {
          deleteDirectory(targetDir + "/" + srcFile)
          moveFileOrDir(srcDir + "/" + srcFile, targetDir)
        } else {
          moveFileOrDir(srcDir + "/" + srcFile, targetDir)
        }
      }
    }
  }
}
/*
object HDFSUtil{
  /* Usage:
  spark-submit --class com.yeti.dwh.edifice.HDFSUtil \
    --master yarn \
    --deploy-mode cluster \
    edw_2.11-1.1.6.jar \
    adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/input \
    adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/processed \
    adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/target \
    adl://yetiadls.azuredatalakestore.net/clusters/data/raw/edifice/archive
  */
  def main(args: Array[String]): Unit = {
    val inputPath = args(0)
    val processedPath = args(1)
    val hiveTablePath = args(2)
    val backupPath = args(3)
    //val loadDateTime = new SimpleDateFormat("yyyy-MM-dd HH:MM:SS").format(Calendar.getInstance().getTime)

    //val rootLogger = Logger.getRootLogger()
    //rootLogger.setLevel(Level.WARN)
    println("<<<delete _SUCCESS file in processed folder " + processedPath)
    if(fsUtil.exists(processedPath + "/" + "_SUCCESS")) {
      fsUtil.deleteDirectory(processedPath + "/" + "_SUCCESS") // delete _SUCCESS file in processed folder
    }
    println("<<<move child directories from " + processedPath + " to " +  hiveTablePath)
    fsUtil.moveNestedDir(processedPath, hiveTablePath) //move child directories from processed folder to target directory
    //fsUtil.moveFileOrDir(inputPath, backupPath + "/loadDateTime=" + loadDateTime) //archive input files after processing for backup
    println("<<<archive input files from " + inputPath + " to " +  backupPath)
    fsUtil.moveInputFiles(inputPath, backupPath) //archive input files after processing for backup
    println("<<<create inputFiles Directory " + inputPath)
    fsUtil.createDirectory(inputPath) //create inputFiles Directory for delta loads
    println(">>>End")
  }

}
*/


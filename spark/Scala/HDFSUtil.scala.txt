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
  var fileSystem: FileSystem = null
  //init() //initialize hadoop fileSystem

  val fs = FileSystem.get(new URI("adl://yetiadls.azuredatalakestore.net"), conf)
  init(fs) //initialize hadoop fileSystem

  def init(fs: FileSystem = null) = {
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
    val srcFiles = listDirectories(srcDir, false)
    for(srcFile <- srcFiles){
      if(exists(srcDir + "/" + srcFile)) {
        val targetFiles = listDirectories(srcDir + "/" + srcFile, false)
        for (targetFile <- targetFiles) {
          if (exists(targetDir + "/" + srcFile + "/" + targetFile)) {
            deleteDirectory(targetDir + "/" + srcFile + "/" + targetFile)
            moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
            //println("1")
            //println(targetDir + "/" + srcFile + "/" + targetFile)
            //println("<<END-1")
          } else {
            //println("2")
            //println(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
            //println("<<END-2")
            if(!exists(targetDir + "/" + srcFile)) {
              createDirectory(targetDir + "/" + srcFile)
            }
            moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
          }
        }
      }
    }
  }

}

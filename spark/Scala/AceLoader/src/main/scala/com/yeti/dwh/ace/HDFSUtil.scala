package com.yeti.dwh.ace

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
    if (null == fs) {
      fileSystem = FileSystem.get(conf)
    } else {
      fileSystem = fs
    }
  }

  //check if the directory exists
  def exists(path: String): Boolean = {
    val p = new Path(path)
    fileSystem.exists(p)
  }

  //drop and recreate the directory
  def recreateDirectory(path: String): Unit = {
    deleteDirectory(path)
    createDirectory(path)
  }

  //delete directory
  def deleteDirectory(path: String): Unit = {
    val dirPath = new Path(path)
    fileSystem.delete(dirPath, true)
  }

  //create directory
  def createDirectory(dirPath: String): Unit = {
    val path = new Path(dirPath)
    if (!fileSystem.exists(path)) {
      fileSystem.mkdirs(path)
    }
  }

  //list child directories with or without absolute path.
  def listDirectories(path: String, fullPath: Boolean): List[String] = {
    var list: List[String] = List[String]()
    val status = fileSystem.listStatus(new Path(path))
    if (fullPath) {
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
    if (fileSystem.exists(srcPath)) {
      fileSystem.rename(srcPath, targetPath)
    }
  }

  def moveNestedDir(srcDir: String, targetDir: String): Unit = {
    val srcFiles = listDirectories(srcDir, fullPath = false)
    for (srcFile <- srcFiles) {
      if (exists(srcDir + "/" + srcFile)) {
        val targetFiles = listDirectories(srcDir + "/" + srcFile, fullPath = false)
        for (targetFile <- targetFiles) {
          if (exists(targetDir + "/" + srcFile + "/" + targetFile)) {
            deleteDirectory(targetDir + "/" + srcFile + "/" + targetFile)
            moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
          } else {
            if (!exists(targetDir + "/" + srcFile)) {
              createDirectory(targetDir + "/" + srcFile)
            }
            moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
          }
        }
      }
    }
  }

  //move ace child directories into target directory
  def moveAceNestedDir(srcDir: String, targetDir: String): Unit = {
    val srcFiles = listDirectories(srcDir, fullPath = false)
    for (srcFile <- srcFiles) {
      val targetFiles = listDirectories(srcDir + "/" + srcFile, fullPath = false)
      deleteDirectory(targetDir + "/" + srcFile) //+ "/" + targetFile)
      for (targetFile <- targetFiles) {
        if (!exists(targetDir + "/" + srcFile)) {
          createDirectory(targetDir + "/" + srcFile)
        }
        moveFileOrDir(srcDir + "/" + srcFile + "/" + targetFile, targetDir + "/" + srcFile)
      }
      deleteDirectory(srcDir + "/" + srcFile)
    }
  }

  //move inputFiles into archive directory
  def moveInputFiles(srcDir: String, targetDir: String): Unit = {
    val srcFiles = listDirectories(srcDir, fullPath = false)
    for (srcFile <- srcFiles) {
      if (exists(srcDir + "/" + srcFile)) {
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

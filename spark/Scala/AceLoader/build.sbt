name := "AceLoader"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-client" % "2.7.0"
)

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.0"


// https://mvnrepository.com/artifact/org.apache.spark/spark-hive
libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.2.0" % "provided"

//artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
//  artifact.name + "_" + sv.full + "-" + "_" + module.revision + "." + artifact.extension
//}

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  "com" + "." + "yeti" + "." + "edw" + "." + "ACE" + "." + artifact.extension
}

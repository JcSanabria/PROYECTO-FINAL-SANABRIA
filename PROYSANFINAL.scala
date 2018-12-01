import org.apache.spark.sql.Encoders
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


case class Inicio(sepalLength: Option[Double], sepalWidth: Option[Double], petalLength: Option[Double], petalWidth: Option[Double], species: Option[String])
case class Ultima(sepalLength : Double, sepalWidth : Double, petalLength : Double, petalWidth : Double, species: Double)


val conf = new SparkConf().setMaster("local[*]").setAppName("IrisSpark")
val sparkSession = SparkSession.builder.config(conf = conf).appName("spark session example").getOrCreate()
val path = "Iris.csv"

var irisSchema2 = Encoders.product[Initial].schema
val iris: DataFrame = sparkSession.read.option("header","true").option("inferSchema", "true").schema(irisSchema2).csv(path)
iris.show()
val assembler = new VectorAssembler().setInputCols(Array("sepalLength", "sepalWidth", "petalLength", "petalWidth", "species")).setOutputCol("features")

def autobot(in: Inicio) = Ultima(
    in.sepalLength.map(_.toDouble).getOrElse(0),
    in.sepalWidth.map(_.toDouble).getOrElse(0),
    in.petalLength.map(_.toDouble).getOrElse(0),
    in.petalWidth.map(_.toDouble).getOrElse(0),
    in.species match {
      case Some("Iris-versicolor") => 1;
      case Some("Iris-virginica") => 2;
      case Some("Iris-setosa") => 3;
      case _ => 3;
    }
  )

val data = assembler.transform(iris.as[Initial].map(autobot))
data.show()


import org.apache.spark.ml.clustering.KMeans


val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(data)


val WSSE = model.computeCost(data)
println(s"Within set sum of Squared Errors = $WSSE")


println("Cluster Centers: ")
model.clusterCenters.foreach(println)

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}


def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))


val bkm = new BisectingKMeans().setK(6)
val model = bkm.run(data)


model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
  println(s"Cluster Center ${idx}: ${center}")
}

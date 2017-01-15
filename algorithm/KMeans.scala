import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data
val data = sc.textFile("/input/logic_noLabel.txt")

val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

// Cluster the data into two classes using KMeans
val numClusters = 3
val numIterations = 20
val numRunTimes = 3
var clusterIndex = 0

val clusters = KMeans.train(parsedData, numClusters, numIterations)

// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)



clusters.clusterCenters.foreach(
  x => {
    println("Center Point of Cluster " + clusterIndex + ":")
    println(x)
    clusterIndex += 1
  })


var clusterNum1 = 0
var clusterNum2 = 0

parsedData.collect().foreach(testDataLine => {
      val predictedClusterIndex:
      Int = clusters.predict(testDataLine)
      println("The data " + testDataLine.toString + " belongs to cluster " +
        predictedClusterIndex)
    })
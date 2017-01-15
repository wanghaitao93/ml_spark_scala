import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

// Load training data in LIBSVM format.
val data = sc.textFile("/input/logic3.txt")

val parsedData = data.map{line=>
val parts = line.split(",")
LabeledPoint(parts(20).toDouble, Vectors.dense(parts.slice(0,20).map(x=>x.toDouble)))
}
// Split the data into training and test sets (30% held out for testing)
val splits = parsedData.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 20 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 10
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val metrics = new MulticlassMetrics(labelAndPreds)

val confuseMetrics = metrics.confusionMatrix

val n11 = labelAndPreds.filter(r => r._1 == 0 && r._2 == 0).count.toDouble
val n22 = labelAndPreds.filter(r => r._1 == 1 && r._2 == 1).count.toDouble
val n12 = labelAndPreds.filter(r => r._1 == 1 && r._2 == 0).count.toDouble
val n21 = labelAndPreds.filter(r => r._1 == 0 && r._2 == 1).count.toDouble

val accuracy = metrics.accuracy
val Sensitivity = n22/(n21 + n22)
val Specificity = n11/(n11 + n12)
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Load training data in LIBSVM format.
val data = sc.textFile("/input/logic3.txt")

val parsedData = data.map{line=>
val parts = line.split(",")
LabeledPoint(parts(20).toDouble, Vectors.dense(parts.slice(0,20).map(x=>x.toDouble)))
}
// Split data into training (60%) and test (40%).
val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)

// Clear the default threshold.
model.clearThreshold()

// Compute raw scores on the test set.
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

val metrics = new MulticlassMetrics(scoreAndLabels)
val accuracy = metrics.accuracy

val confuseMetrics = metrics.confusionMatrix




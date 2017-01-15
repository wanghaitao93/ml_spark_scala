import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val data = sc.textFile("/input/logic3.txt")

val parsedData = data.map{line=>
val parts = line.split(",")
LabeledPoint(parts(20).toDouble, Vectors.dense(parts.slice(0,20).map(x=>x.toDouble)))
}

val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 9L)
val trainingData = splits(0)
val testData = splits(1)

val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)

val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count

val metrics = new MulticlassMetrics(labelAndPreds)
val confuseMetrics = metrics.confusionMatrix


val n11 = labelAndPreds.filter(r => r._1 == 0 && r._2 == 0).count.toDouble
val n22 = labelAndPreds.filter(r => r._1 == 1 && r._2 == 1).count.toDouble
val n12 = labelAndPreds.filter(r => r._1 == 1 && r._2 == 0).count.toDouble
val n21 = labelAndPreds.filter(r => r._1 == 0 && r._2 == 1).count.toDouble

val accuracy = metrics.accuracy
val Sensitivity = n22/(n21 + n22)
val Specificity = n11/(n11 + n12)


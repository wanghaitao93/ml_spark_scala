import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.mllib.evaluation.MulticlassMetrics



val data = spark.read.format("libsvm").load("/input/sample_NN.txt").toDF()

// Split the data into train and test
val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](20, 5, 4, 2)

// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// train the model
val model = trainer.fit(train)

// compute accuracy on the test set
val result = model.transform(test)
// val predictionAndLabels = result.select("label", "prediction")

val predictionAndLabels =
result.select(col("prediction"), col("label").cast(DoubleType)).rdd.map {
  case Row(prediction: Double, label: Double) => (prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)
val confuseMetrics = metrics.confusionMatrix

val n11 = predictionAndLabels.filter(r => r._1 == 0 && r._2 == 0).count.toDouble
val n22 = predictionAndLabels.filter(r => r._1 == 1 && r._2 == 1).count.toDouble
val n12 = predictionAndLabels.filter(r => r._1 == 1 && r._2 == 0).count.toDouble
val n21 = predictionAndLabels.filter(r => r._1 == 0 && r._2 == 1).count.toDouble

val accuracy = metrics.accuracy
val Sensitivity = n22/(n21 + n22)
val Specificity = n11/(n11 + n12)

// val evaluator = new MulticlassClassificationEvaluator().setMetricName("precision")
// println("Precision:" + evaluator.evaluate(predictionAndLabels))



from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer

# Start Spark session
spark = SparkSession.builder \
    .appName("AdvancedModels") \
    .getOrCreate()

print("Loading data...")
train_df = spark.read.parquet("data/processed/train_data.parquet")
val_df = spark.read.parquet("data/processed/val_data.parquet")

print("Fixing feature data types...")
to_vector_udf = udf(lambda a: Vectors.dense(a), VectorUDT())
train_df = train_df.withColumn("features_vec", to_vector_udf("tfidf_features"))
val_df = val_df.withColumn("features_vec", to_vector_udf("tfidf_features"))

# ---------------------------------------------------------
# THE FIX: Create numeric 'label' column from 'method' column
# ---------------------------------------------------------
print("Creating numeric label column...")
indexer = StringIndexer(inputCol="method", outputCol="label", handleInvalid="keep")
indexer_model = indexer.fit(train_df)
train_df = indexer_model.transform(train_df)
val_df = indexer_model.transform(val_df)

# Setup evaluators
eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# ---------------------------------------------------------
# 1. Random Forest Classifier
# ---------------------------------------------------------
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(featuresCol="features_vec", labelCol="label", numTrees=20)
rf_model = rf.fit(train_df)

print("Evaluating Random Forest...")
rf_preds = rf_model.transform(val_df)
print(f"Random Forest Accuracy: {eval_acc.evaluate(rf_preds):.4f}")
print(f"Random Forest F1-Score: {eval_f1.evaluate(rf_preds):.4f}")

print("Saving Random Forest model...")
rf_model.write().overwrite().save("models/rf_model")

# ---------------------------------------------------------
# 2. Linear Support Vector Machine (SVM) using OneVsRest
# ---------------------------------------------------------
print("\nTraining Linear SVM model...")
svm = LinearSVC(featuresCol="features_vec", maxIter=10)
# Wrapping SVM in OneVsRest to support multi-class target
ovr = OneVsRest(classifier=svm, featuresCol="features_vec", labelCol="label")
svm_model = ovr.fit(train_df)

print("Evaluating Linear SVM...")
svm_preds = svm_model.transform(val_df)
print(f"Linear SVM Accuracy: {eval_acc.evaluate(svm_preds):.4f}")
print(f"Linear SVM F1-Score: {eval_f1.evaluate(svm_preds):.4f}")

print("Saving Linear SVM model...")
svm_model.write().overwrite().save("models/svm_model")

print("\nTask 3.5 Completed!")
spark.stop()

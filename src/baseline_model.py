from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("BaselineModel_LogisticRegression") \
    .getOrCreate()

# 2. Load the training and validation datasets
print("Loading training and validation data...")
train_df = spark.read.parquet("data/processed/train_data.parquet")
val_df = spark.read.parquet("data/processed/val_data.parquet")

# 3. Initialize the Logistic Regression model (Baseline)
# Assuming your label column is named 'label' and features column is 'features'
print("Initializing Logistic Regression model...")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# 4. Train the model using the training data
print("Training the baseline model (this might take a minute)...")
lr_model = lr.fit(train_df)

# 5. Make predictions on the validation data
print("Evaluating model on validation data...")
predictions = lr_model.transform(val_df)

# 6. Calculate Accuracy and F1-Score
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# 7. Print the results
print("-" * 40)
print("Baseline Model Results (Logistic Regression):")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print("-" * 40)

# Save the trained model for future use
print("Saving the model...")
lr_model.write().overwrite().save("models/baseline_lr_model")

print("Task 3.4 Completed!")
spark.stop()

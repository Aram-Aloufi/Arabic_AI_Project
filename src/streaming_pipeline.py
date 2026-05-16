from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

# Start Spark session
spark = SparkSession.builder \
    .appName("RealTimeDetectionStream") \
    .getOrCreate()

# Reduce log spam
spark.sparkContext.setLogLevel("WARN")

# Load the trained Random Forest model
print("Loading Random Forest model...")
rf_model = RandomForestClassificationModel.load("models/rf_model")

# Read schema from test data
print("Preparing stream schema...")
schema = spark.read.parquet("data/processed/test_data.parquet").schema

# Task 4.1: Stream Simulation - monitor the input folder
print("Monitoring 'data/streaming_input'...")
streaming_df = spark.readStream \
    .schema(schema) \
    .parquet("data/streaming_input")

# THE FIX: Convert tfidf_features to features_vec just like we did in training
to_vector_udf = udf(lambda a: Vectors.dense(a), VectorUDT())
streaming_df = streaming_df.withColumn("features_vec", to_vector_udf("tfidf_features"))

# Task 4.2: Real-Time Deployment - make predictions
predictions = rf_model.transform(streaming_df)

# Select specific columns to display in console
output_df = predictions.select("method", "prediction")

# Output stream to the console
print("Stream is ACTIVE! Waiting for files... (Press Ctrl+C to stop)")
query = output_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()

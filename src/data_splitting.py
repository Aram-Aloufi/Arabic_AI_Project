from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder \
    .appName("DataSplitting") \
    .getOrCreate()

# Load the processed final ML dataset
print("Loading data...")
df = spark.read.parquet("data/processed/final_ml_dataset.parquet")

# Split data: 70% train, 15% validation, 15% test
# Using seed=42 to ensure exact same split every time
print("Splitting data...")
train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)

# Print row counts to verify
print("Total rows:", df.count())
print("Train rows:", train_df.count())
print("Validation rows:", val_df.count())
print("Test rows:", test_df.count())

# Save the splits to new parquet files inside data/processed directory
print("Saving files...")
train_df.write.mode("overwrite").parquet("data/processed/train_data.parquet")
val_df.write.mode("overwrite").parquet("data/processed/val_data.parquet")
test_df.write.mode("overwrite").parquet("data/processed/test_data.parquet")

print("Data splitting completed!")
spark.stop()

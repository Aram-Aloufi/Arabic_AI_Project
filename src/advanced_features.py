from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.functions import col

print("Starting Spark session for Advanced Feature Engineering...")
spark = SparkSession.builder \
    .appName("AdvancedFeatures_TFIDF") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Loading Task 3.1 features dataset...")
df = spark.read.parquet("data/processed/features_dataset.parquet")

# Ensure we use the fully cleaned text (no diacritics, no stopwords)
clean_text_col = "fully_cleaned_abstract"
df_clean = df.filter(col(clean_text_col).isNotNull() & (col(clean_text_col) != ""))

print("Tokenizing text...")
tokenizer = Tokenizer(inputCol=clean_text_col, outputCol="words")
wordsData = tokenizer.transform(df_clean)

print("Applying HashingTF (Term Frequency)...")
# We use 5000 features to balance representation richness and memory usage
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)
featurizedData = hashingTF.transform(wordsData)

print("Fitting IDF (Inverse Document Frequency) model...")
idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Drop intermediate lists to keep the dataframe clean and efficient
final_df = rescaledData.drop("words", "rawFeatures")

output_path = "data/processed/final_ml_dataset.parquet"
print(f"Saving final ML dataset to Parquet format...")
final_df.write.mode("overwrite").parquet(output_path)

print("Advanced Feature Engineering (Task 3.2) completed successfully.")
spark.stop()

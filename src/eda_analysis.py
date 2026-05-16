from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, desc
from pyspark.ml.feature import NGram

print("Starting Spark session for EDA...")
spark = SparkSession.builder \
    .appName("EDA_Analysis") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Loading processed Parquet data...")
df = spark.read.parquet("data/processed/arabic_abstracts.parquet")

# Filter out empty or null abstracts
df_clean = df.filter(col("fully_cleaned_abstract").isNotNull() & (col("fully_cleaned_abstract") != ""))

print("\n--- 1. Word Frequency (For Word Cloud) ---")
# Split text into words and explode into rows
words_df = df_clean.withColumn("word", explode(split(col("fully_cleaned_abstract"), " ")))
words_df = words_df.filter(col("word") != "")

# Count word frequencies
word_counts = words_df.groupBy("word").count().orderBy(desc("count"))
print("Top 15 most frequent words:")
word_counts.show(15, truncate=False)

print("\n--- 2. N-Gram Frequency (Bi-grams) ---")
# Tokenize into arrays for NGram function
tokenized_df = df_clean.withColumn("tokens", split(col("fully_cleaned_abstract"), " "))
ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
bigram_df = ngram.transform(tokenized_df)

# Explode bigrams and count
bigrams_exploded = bigram_df.withColumn("bigram", explode(col("bigrams")))
bigram_counts = bigrams_exploded.groupBy("bigram").count().orderBy(desc("count"))
print("Top 10 most frequent bi-grams:")
bigram_counts.show(10, truncate=False)

print("\n--- 3. Vocabulary Richness (Type-Token Ratio - TTR) ---")
total_tokens = words_df.count()
total_types = word_counts.count()

if total_tokens > 0:
    ttr = total_types / total_tokens
else:
    ttr = 0.0

print(f"Total Tokens (All Words): {total_tokens}")
print(f"Total Types (Unique Words): {total_types}")
print(f"Type-Token Ratio (TTR): {ttr:.4f}")

print("\nSaving top words to CSV for Word Cloud generation...")
word_counts.limit(100).coalesce(1).write.mode("overwrite").csv("data/processed/top_words_for_cloud", header=True)

print("EDA completed successfully.")
spark.stop()

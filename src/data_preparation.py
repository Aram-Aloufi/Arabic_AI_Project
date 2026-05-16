from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import pyarabic.araby as araby
from tashaphyne.stemming import ArabicLightStemmer
import nltk

# Download stopwords quietly
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ArabicLightStemmer()

# Function to remove diacritics only
def normalize_text(text):
    if text is None: return ""
    return araby.strip_tashkeel(text)

# Function for full cleaning: diacritics, stopwords, and stemming
def clean_and_stem(text):
    if text is None: return ""
    text = araby.strip_tashkeel(text)
    words = araby.tokenize(text)
    cleaned_words = [stemmer.light_stem(w) for w in words if w not in arabic_stopwords]
    return " ".join(cleaned_words)

# Register Spark UDFs
normalize_udf = udf(normalize_text, StringType())
clean_stem_udf = udf(clean_and_stem, StringType())

print("Starting Spark session...")
spark = SparkSession.builder \
    .appName("DataPreprocessing") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Loading dataset...")
df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

# Find the text column
text_col = df.columns[0]
for c in df.columns:
    if "abstract" in c.lower() or "text" in c.lower():
        text_col = c
        break

print("Applying preprocessing pipeline...")
# Create new columns for different stages of cleaning
processed_df = df.withColumn("normalized_abstract", normalize_udf(col(text_col))) \
                 .withColumn("fully_cleaned_abstract", clean_stem_udf(col(text_col)))

output_path = "data/processed/arabic_abstracts.parquet"
print("Saving data to Parquet format...")
processed_df.write.mode("overwrite").parquet(output_path)

print("Processing completed successfully.")
spark.stop()

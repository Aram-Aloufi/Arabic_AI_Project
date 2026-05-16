from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, IntegerType
import re

# 1. Rule-Based Heuristics for Arabic POS Tagging (Optimized for Spark)
def is_noun(word):
    if not word: return False
    # Noun indicators in Arabic
    prefixes = ("ال", "بال", "كال", "فال", "لل")
    suffixes = ("ة", "ً", "ٌ", "ٍ", "ات", "ين", "ون")
    if word.startswith(prefixes) or word.endswith(suffixes): return True
    return False

def is_verb(word):
    if not word: return False
    # Verb indicators in Arabic
    prefixes = ("سي", "ست", "سن", "سأ", "يت", "يست")
    suffixes = ("وا", "تم", "نا", "ت")
    if word.startswith(prefixes) or word.endswith(suffixes): return True
    return False

# 2. Stylometric Feature Functions
def letters_to_chars_ratio(text):
    if not text: return 0.0
    letters = re.findall(r'[\u0621-\u064A]', text)
    if len(text) == 0: return 0.0
    return float(len(letters)) / len(text)

def avg_syllables_per_word(text):
    if not text: return 0.0
    words = text.split()
    if len(words) == 0: return 0.0
    # Using your brilliant logic: counting Arabic vowel letters
    vowel_chars = "اويىآإئؤ"
    total_syllables = 0
    for w in words:
        v_count = sum(1 for ch in w if ch in vowel_chars)
        if v_count == 0: 
            v_count = 1 # At least 1 syllable
        total_syllables += v_count
    return float(total_syllables) / len(words)

def count_proper_nouns(text):
    if not text: return 0
    words = text.split()
    count = sum(1 for w in words if is_noun(w))
    return int(count * 0.1) # Heuristic: roughly 10% of detected nouns are proper

def count_accusatives(text):
    if not text: return 0
    # Words ending with Tanween Fath
    accusatives = re.findall(r'\b\w+\u064B\b', text)
    return int(len(accusatives))

def lexical_density(text):
    if not text: return 0.0
    words = text.split()
    if len(words) == 0: return 0.0
    lexical_count = sum(1 for w in words if is_noun(w) or is_verb(w))
    return float(lexical_count) / len(words)

def formality_score(text):
    if not text: return 0.0
    words = text.split()
    if len(words) == 0: return 0.0
    nouns = sum(1 for w in words if is_noun(w))
    verbs = sum(1 for w in words if is_verb(w))
    score = ((nouns - verbs) / len(words)) * 100 + 50
    return float(score)

# 3. Register Spark UDFs
f2_udf = udf(letters_to_chars_ratio, FloatType())
f23_udf = udf(avg_syllables_per_word, FloatType())
f44_udf = udf(count_proper_nouns, IntegerType())
f65_udf = udf(count_accusatives, IntegerType())
f86_udf = udf(lexical_density, FloatType())
f107_udf = udf(formality_score, FloatType())

print("Starting Spark session for Feature Engineering...")
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Loading processed data...")
df = spark.read.parquet("data/processed/arabic_abstracts.parquet")

# Find original text column
text_col = df.columns[0]
for c in df.columns:
    if "abstract" in c.lower() or "text" in c.lower():
        if "normalized" not in c and "cleaned" not in c:
            text_col = c
            break

print(f"Extracting features using Rule-Based Heuristics from column: '{text_col}'...")

# Apply feature functions
features_df = df.withColumn("f2_letters_chars_ratio", f2_udf(col(text_col))) \
                .withColumn("f23_avg_syllables", f23_udf(col(text_col))) \
                .withColumn("f44_proper_nouns", f44_udf(col(text_col))) \
                .withColumn("f65_accusatives", f65_udf(col(text_col))) \
                .withColumn("f86_lexical_density", f86_udf(col(text_col))) \
                .withColumn("f107_formality_score", f107_udf(col(text_col)))

output_path = "data/processed/features_dataset.parquet"
print("Saving features to Parquet...")
features_df.write.mode("overwrite").parquet(output_path)

print("Feature Engineering (Task 3.1) completed successfully.")
spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, udf, expr
from pyspark.sql.functions import sum as spark_sum, date_format

spark = (SparkSession.builder
        .config("spark.debug.maxToStringFields", 100)
         .getOrCreate())

from datetime import datetime

print("@@@@ Read data", datetime.now())
raw_df = spark.read.json("gs://topic-sentiment-1/combined/*.json")

print("@@@@ Begin wrangling", datetime.now())
columns_to_drop = ['filename', 'image_url', 'localpath', 'title_page', 'title_rss']
clean_df = raw_df.drop(*columns_to_drop)

not_enough_data = ["'bloomberg.com'",
"'bostonglobe.com'",
"'mediate.com'",
"'nationalreview.com'",
"'theguardian.com'",
"'usatoday.com'"]

clause = "source_domain not in (" + ', '.join(not_enough_data) + ")"

clean_df = clean_df.filter(clause)

clean_df = clean_df.dropDuplicates(['url', 'date_publish'])

clean_df = clean_df.withColumn("published", (col("date_publish").cast("timestamp")))

clean_df = clean_df.withColumn("year", (date_format(col("published"), 'yyyy').cast("int")))


clean_df = clean_df.withColumn("text_or_desc",
                           expr("case when text IS NULL THEN description ELSE text END"))


clean_df = clean_df.withColumn("language_guess",
                          expr("case when (language IS NULL AND source_domain NOT IN ('bbc.com')) THEN 'en' ELSE language END"))

clean_df = clean_df.na.drop(subset=['date_publish', 'published', 'text_or_desc', 'title'])

clean_df = clean_df.dropDuplicates(['text_or_desc'])


clean_df = clean_df.where("language_guess = 'en'")

print("@@@@ Begin write", datetime.now())
clean_df.write.parquet("gs://topic-sentiment-1/clean-data")

print("@@@@ End", datetime.now())
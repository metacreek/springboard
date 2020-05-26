from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, udf, expr, date_format, rank
from pyspark.sql.window import Window

spark = (SparkSession.builder
        .config("spark.debug.maxToStringFields", 100)
        .getOrCreate()
         )

RAW_DATA_INPUT = '"gs://topic-sentiment-1/raw_data_test/sample3.json"'
#WRANGLED_DATA_OUTPUT =

raw_df = spark.read.json(RAW_DATA_INPUT)

# delete columns we don't need
columns_to_drop = ['filename', 'image_url', 'localpath', 'title_page', 'title_rss']
clean_df = raw_df.drop(*columns_to_drop)

# drop duplicates
clean_df = clean_df.dropDuplicates(['url', 'date_publish'])

# create timestamp version of string date_publish
clean_df = clean_df.withColumn("published", (col("date_publish").cast("timestamp")))

# Useful in cleaning and analysis
clean_df = clean_df.withColumn("year", (date_format(col("published"), 'yyyy').cast("int")))
clean_df = clean_df.withColumn("month", (date_format(col("published"), 'M').cast("int")))

# We create a new column text_or_desc that will be used in analysis.
# This uses the text column data, if present, and falls back to description if the text column was empty.
clean_df = clean_df.withColumn("text_or_desc",
                           expr("case when text IS NULL THEN description ELSE text END"))

# we assume that the language is English (en) if the language was not specified.
# We are excluding bbc.com where this was not a good assumption.
# See EDA-inital notebook for more info.
clean_df = clean_df.withColumn("language_guess",
                          expr("case when (language IS NULL AND source_domain NOT IN ('bbc.com')) THEN 'en' ELSE language END"))

# We are only doing english
clean_df = clean_df.where("language_guess = 'en'")

# get rid of any rows where the date of publish, the title or the text/description is empty.
clean_df = clean_df.na.drop(subset=['date_publish', 'published', 'text_or_desc', 'title'])

# additional duplicate removal
clean_df = clean_df.dropDuplicates(['text_or_desc'])

# Use only data since 2010
clean_df = clean_df.where('year >= 2010')

# Make no publication have more than 3.5 percent of the data
subtotal = clean_df.count()
upper_threshold = 0.035 * subtotal
# see https://stackoverflow.com/a/38398563/914544
window = Window.partitionBy(clean_df['source_domain']).orderBy(clean_df['published'].desc())
clean_df = clean_df.select('*', rank().over(window).alias('rank')).filter(col('rank') <= upper_threshold)

# Get rid of publications with small data
subtotal = clean_df.count()
lower_threshold = 0.005 * subtotal

total_by_publication = clean_df.group_by('source_domain').count()
total_by_publication = total_by_publication.where(f'count > {lower_threshold}')


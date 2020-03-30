from pyspark.sql import SparkSession
import pyspark.sql.functions as F


VER = '10percent-v1'

spark = (SparkSession.builder
        .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

VER = '10percent-v1'

features_sdf = spark.read.parquet(f'gs://topic-sentiment-1/features/{VER}')

features_sdf = features_sdf.withColumn('concat', F.concat_ws(' ', 'named_entities'))

selected_sdf = features_sdf.filter(features_sdf.concat.contains('Trump'))

selected_sdf = selected_sdf.select('published_date', 'weeks', 'source_domain',
                                   'sentiment_direction', 'sentiment_confidence')

selected_sdf.write.csv('gs://topic-sentiment-1/plot/trump')
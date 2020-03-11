from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when
from pyspark.ml.feature import CountVectorizer, IDF

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import SentenceDetector, Tokenizer, LemmatizerModel, StopWordsCleaner, \
    Normalizer, WordEmbeddingsModel, NerDLModel, NerConverter, ViveknSentimentModel

import contractions

VER = '1percent-v1'


from datetime import datetime

spark = (SparkSession.builder
         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0")
         .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

# load wrangled data
print("@@@@ Begin data read", datetime.now())
raw_sdf = spark.read.json('gs://topic-sentiment-1/sample2.json')

print("@@@@ Begin repartition", datetime.now())
raw_sdf = raw_sdf.repartition(20)

print("Data size", raw_sdf.count())

print("Partition size", raw_sdf.rdd.getNumPartitions())

# add id column
print("@@@@ Begin pre-processing", datetime.now())
processed_sdf = raw_sdf.select("*").withColumn("id", F.monotonically_increasing_id())

# simple fix to contractions
contractions_udf = F.udf(contractions.fix)
processed_sdf = processed_sdf.withColumn('clean_text', contractions_udf('text_or_desc'))

# remove possessives
processed_sdf = processed_sdf.withColumn('clean_text', F.regexp_replace('clean_text', "'s", ""))

# remove non text
processed_sdf = processed_sdf.withColumn('clean_text', F.regexp_replace('clean_text', "[^A-Za-z. \n]", ""))

# standardize US
processed_sdf = processed_sdf.withColumn('clean_text', F.regexp_replace('clean_text', "(US|U.S.|USA)", "United States"))

# date normalization
processed_sdf = processed_sdf.withColumn('published_date', F.to_date(col('published')))
processed_sdf = processed_sdf.withColumn('start_date', F.lit('2016-01-01'))
processed_sdf = processed_sdf.withColumn('weeks', F.floor(F.datediff(col('published'), F.lit('2016-01-01'))/7))

# Begin pipeline
print("@@@@ Begin pipeline", datetime.now())
documentAssembler = DocumentAssembler().setInputCol("clean_text").setOutputCol("document")

sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

lemma = (LemmatizerModel.pretrained('lemma_antbnc')
         .setInputCols(['token']).setOutputCol('lemma'))

stop_words_cleaner = StopWordsCleaner().setInputCols(['lemma']).setOutputCol('stopless')
stop_words = stop_words_cleaner.getStopWords()
stop_words.remove('not')
stop_words.remove('cannot')
stop_words.remove('against')
stop_words.remove('nor')
stop_words.remove('no')
stop_words = [word.replace("'", "") for word in stop_words]
stop_words_cleaner.setStopWords(stop_words)

normalizer = Normalizer().setInputCols(["stopless"]).setOutputCol("normalized")

word_embed = (WordEmbeddingsModel.pretrained()
              .setInputCols(['sentence', 'stopless'])
              .setOutputCol('embedding')
              )

ner_dl = (NerDLModel().pretrained()
          .setInputCols(['document', 'stopless', 'embedding'])
          .setOutputCol('ner')
          )

ner_conv = (NerConverter()
            .setInputCols(["document", "normalized", "ner"])
            .setOutputCol('ner_converted')
            )

sentiment_model = (ViveknSentimentModel.pretrained()
                   .setInputCols(['document', 'token'])
                   .setOutputCol('sentiment')
                   )

finisher = Finisher().setInputCols(
    ["sentence", "token", "lemma", "stopless", "embedding",
     "normalized", "ner", "ner_converted",
     'sentiment']).setCleanAnnotations(True)

pipeline = (Pipeline()
            .setStages([
                        documentAssembler,
                        sentenceDetector,
                        tokenizer,
                        lemma,
                        stop_words_cleaner,
                        normalizer,
                        word_embed,
                        ner_dl,
                        ner_conv,
                        sentiment_model,
                        finisher
                        ]))

print("@@@@ Begin pipeline fit", datetime.now())
pipeline_model = pipeline.fit(processed_sdf)

processed_sdf = pipeline_model.transform(processed_sdf)

processed_sdf = (processed_sdf
                 .withColumn('sentiment',
                             when(col('finished_sentiment')[0] == 'positive', 1).when(
                                 col('finished_sentiment')[0] == 'negative', -1).otherwise(0)
                             ))

print("@@@@ Begin count vectorizer", datetime.now())
cv = CountVectorizer(inputCol="finished_ner_converted",
                     outputCol="ner_vectors",
                     maxDF=0.75,
                     minDF=25
                     )

cv_model = cv.fit(processed_sdf)

processed_sdf = cv_model.transform(processed_sdf)

print("@@@@ Begin IDF", datetime.now())
idf = IDF(inputCol="ner_vectors", outputCol="ner_vectors_idf")

idf_model = idf.fit(processed_sdf)

processed_sdf = idf_model.transform(processed_sdf)

print("Partition size", processed_sdf.rdd.getNumPartitions())

print("@@@@ Writing results", datetime.now())

print("Partition size", processed_sdf.rdd.getNumPartitions())

processed_sdf.write.parquet(f'gs://topic-sentiment-1/features/{VER}')

print("@@@@ Saving CV Model", datetime.now())

cv_model.save(f'gs://topic-sentiment-1/vocab_model/{VER}')

print("@@@@ Finished", datetime.now())

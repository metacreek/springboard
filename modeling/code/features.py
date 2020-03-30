from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when
from pyspark.ml.feature import CountVectorizer, IDF, HashingTF
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StringType

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, Tokenizer, LemmatizerModel, StopWordsCleaner, \
    Normalizer, WordEmbeddingsModel, NerDLModel, NerConverter, ViveknSentimentModel, PerceptronModel

import contractions

VER = '10percent-v1'


from datetime import datetime

spark = (SparkSession.builder
         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")
         .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

# load wrangled data
print("@@@@ Begin data read", datetime.now())
raw_sdf = spark.read.parquet('gs://topic-sentiment-1/decade-data/ten-percent')

print("@@@@ Begin repartition", datetime.now())
raw_sdf = raw_sdf.repartition(60)

print("Data size", raw_sdf.count())

print("Partition size", raw_sdf.rdd.getNumPartitions())

# add id column
print("@@@@ Begin pre-processing", datetime.now())
processed_sdf = raw_sdf.select("*").withColumn("id", F.monotonically_increasing_id())

# date normalization
processed_sdf = processed_sdf.withColumn('published_date', F.to_date(col('published')))
processed_sdf = processed_sdf.withColumn('start_date', F.lit('2010-01-01'))
processed_sdf = processed_sdf.withColumn('weeks', F.floor(F.datediff(col('published'), F.lit('2016-01-01'))/7))

# simple fix to contractions
contractions_udf = F.udf(contractions.fix)
processed_sdf = processed_sdf.withColumn('clean_text', contractions_udf('text_or_desc'))

# remove non text
processed_sdf = processed_sdf.withColumn('clean_text', F.regexp_replace('clean_text', "[^A-Za-z.\n]", " "))

# standardize US
processed_sdf = processed_sdf.withColumn('clean_text', F.regexp_replace('clean_text', "(US|U.S.|USA)", "United States"))

# Begin pipeline
print("@@@@ Begin pipeline", datetime.now())
documentAssembler = DocumentAssembler().setInputCol("clean_text").setOutputCol("document")

sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

lemma = (LemmatizerModel.pretrained('lemma_antbnc')
         .setInputCols(['token']).setOutputCol('lemma'))

pub_names = ['NPR', 'AP', 'CNN', 'Reuters', 'Times', 'New York Times', 'Nation Travels', 'Nation',
             'Getty', 'Advertisement', 'Fox News', 'Fox', 'NBC', 'Washington Post', 'Post',
             'Associated Press', 'Associated', 'Hill', 'Support Progressive Journalism Nation',
             'CBS', 'RT', 'Daily Dispatch', 'Dispatch', 'Slate', 'Like', 'ADVERTISEMENT',
             'Business Wire', 'Wire', 'MSNBC', 'ABC', 'Atlantic', 'Privacy Policy Sign Take Action',
             'Hill K Street NW Suite', 'Capitol Hill Publishing Corp', 'Publishing', 'Corp',
             'News Communications Inc', 'Inc', 'Communications', 'Subscribe', 'Wall Street Journal',
             'Journal', 'BBC', 'New York Times Archives', 'Archives', 'Politico', 'CNBC',
             'Newsday', 'Vox', 'CBS News', 'Privacy Policy Sign Take', 'Ad', 'Enlarge',
             'REUTERS', 'Los Angeles Times', 'NBC News', 'Breitbart', 'FOX NEWS APP',
             'ABC News', 'Back Gallery', 'New Republic', 'Breitbart News', 'Page Buy Reprints',
             'Reprints', 'Atlanta Journal Constitution', 'Time', 'LA Times', 'Axios',
             'Newsweek', 'WSJ', 'TheBlaze', 'Boston Globe', 'Globe', 'unitName dfpPosition',
             'unitName', 'dfpPosition', 'National Review', 'fromMarch', 'Denver Post',
             'Subscribe Newsday', 'Subscribe', 'Journal Constitution', 'CHRON',
             'Daily Beast', 'Beast', 'Sun Times', 'Financial Times', 'New York Post',
             'Copyright', 'Atlanta Journal', 'PHOTO', 'Images', 'ResearchAndMarketscom',
             'Full Post', 'Current Issue View'
             ]

stop_words_cleaner = StopWordsCleaner().setInputCols(['lemma']).setOutputCol('stopless')
stop_words = stop_words_cleaner.getStopWords()
stop_words.remove('not')
stop_words.remove('cannot')
stop_words.remove('against')
stop_words.remove('nor')
stop_words.remove('no')
for words in pub_names:
    stop_words.append(words)
stop_words = [word.replace("'", "") for word in stop_words]
stop_words_cleaner.setStopWords(stop_words)

normalizer = Normalizer().setInputCols(["stopless"]).setOutputCol("normalized")

word_embed = (WordEmbeddingsModel.pretrained()
              .setInputCols(['sentence', 'normalized'])
              .setOutputCol('glove_embedding')
              )

ner_dl = (NerDLModel().pretrained()
          .setInputCols(['document', 'normalized', 'glove_embedding'])
          .setOutputCol('ner')
          )

ner_conv = (NerConverter()
            .setInputCols(["document", "normalized", "ner"])
            .setOutputCol('ner_converted')
            )

sentiment_model = (ViveknSentimentModel.pretrained()
                   .setInputCols(['normalized', 'document'])
                   .setOutputCol('sentiment')
                   )

pos_tagger = (PerceptronModel.pretrained()
              .setInputCols(['normalized', 'document'])
              .setOutputCol('parts_of_speech')
             )

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
                        pos_tagger
                        ]))

print("@@@@ Begin pipeline fit", datetime.now())
pipeline_model = pipeline.fit(processed_sdf)

processed_sdf = pipeline_model.transform(processed_sdf)

# convert sentiment into usable values
def process_sentiment_direction(col):
    direction = 0
    for row in col:
        if row.result == 'positive':
            direction = 1
        elif row.result == 'negative':
            direction = -1
    return direction


udf_process_sentiment_direction = F.udf(process_sentiment_direction, IntegerType())


def process_sentiment_confidence(col):
    direction = 0.0
    for row in col:
        if row.result == 'positive':
            direction = float(row.metadata['confidence'])
        elif row.result == 'negative':
            direction = -1.0 * float(row.metadata['confidence'])
    return direction


udf_process_sentiment_confidence = F.udf(process_sentiment_confidence, FloatType())

processed_sdf = processed_sdf.withColumn('sentiment_direction', udf_process_sentiment_direction('sentiment'))
processed_sdf = processed_sdf.withColumn('sentiment_confidence', udf_process_sentiment_confidence('sentiment'))


def process_nouns(col):
    nouns = []
    for row in col:
        if row.result in ['NN', 'NNS']:
            nouns.append(row.metadata['word'])
    return nouns


udf_process_nouns = F.udf(process_nouns, ArrayType(StringType()))

processed_sdf = processed_sdf.withColumn('nouns', udf_process_nouns('parts_of_speech'))


def process_ner(col):
    nouns = []
    last_type = 'O'
    for row in col:
        if row.result != 'O':
            if last_type == row.result:
                last = nouns.pop(-1)
                new_word = f"{last} {row.metadata['word']}"
                nouns.append(new_word)
            else:
                nouns.append(row.metadata['word'])
        last_type = row.result
    return nouns


udf_process_ner = F.udf(process_ner, ArrayType(StringType()))

processed_sdf = processed_sdf.withColumn('named_entities', udf_process_ner('ner'))

#reduce the data we are carrying around
processed_sdf = processed_sdf.select('published', 'source_domain', 'weeks', 'title', 'url', 'year', 'id',
                                     'published_date', 'sentiment_direction', 'glove_embedding',
                                     'sentiment_confidence', 'nouns', 'named_entities')

print("@@@@ Begin count vectorizer", datetime.now())
cv = CountVectorizer(inputCol="named_entities",
                     outputCol="ner_vectors",
                     maxDF=0.8,
                     minDF=25,
                     vocabSize=2048
                     )

cv_model = cv.fit(processed_sdf)

processed_sdf = cv_model.transform(processed_sdf)

print("@@@@ Begin IDF", datetime.now())
idf = IDF(inputCol="ner_vectors", outputCol="ner_vectors_idf")

idf_model = idf.fit(processed_sdf)

processed_sdf = idf_model.transform(processed_sdf)

print("@@@@ Begin noun processing", datetime.now())

hashing_tf = HashingTF(inputCol="nouns", outputCol="noun_vectors", numFeatures=2048)

processed_sdf = hashing_tf.transform(processed_sdf)

noun_idf = IDF(inputCol="noun_vectors", outputCol="noun_vectors_idf")

noun_idf_model = noun_idf.fit(processed_sdf)

processed_sdf = noun_idf_model.transform(processed_sdf)


print("Partition size", processed_sdf.rdd.getNumPartitions())

print("@@@@ Writing results", datetime.now())

print("Partition size", processed_sdf.rdd.getNumPartitions())

processed_sdf.write.parquet(f'gs://topic-sentiment-1/features/{VER}')

print("@@@@ Saving CV Model", datetime.now())

cv_model.save(f'gs://topic-sentiment-1/vocab_model/{VER}')

print("@@@@ Finished", datetime.now())

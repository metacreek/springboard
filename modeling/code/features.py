from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, monotonically_increasing_id, udf, regexp_replace, \
    floor, to_date, datediff, lit
from pyspark.ml.feature import CountVectorizer, IDF, HashingTF
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, Tokenizer, LemmatizerModel, StopWordsCleaner, \
    Normalizer, WordEmbeddingsModel, NerDLModel, NerConverter, ViveknSentimentModel, \
    PerceptronModel, BertEmbeddings, SentenceEmbeddings
import contractions
from datetime import datetime

VER = 'sentence-1percent-v2'

def log_time_message(msg):
    print(f"@@@@ {msg}", datetime.now())


def process_sentiment_direction(col):
    direction = 0
    for row in col:
        if row.result == 'positive':
            direction = 1
        elif row.result == 'negative':
            direction = -1
    return direction


def process_sentiment_confidence(col):
    direction = 0.0
    for row in col:
        if row.result == 'positive':
            direction = float(row.metadata['confidence'])
        elif row.result == 'negative':
            direction = -1.0 * float(row.metadata['confidence'])
    return direction


def process_nouns(col):
    nouns = []
    for row in col:
        if row.result in ['NN', 'NNS']:
            nouns.append(row.metadata['word'])
    return nouns


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

spark = (SparkSession.builder
         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")
         .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

# load wrangled data
log_time_message("Begin data read")
raw_sdf = spark.read.json('gs://topic-sentiment-1/sample2.json')

log_time_message("Begin repartition")
raw_sdf = raw_sdf.repartition(20)

print("Data size", raw_sdf.count())

print("Partition size", raw_sdf.rdd.getNumPartitions())

# add id column
log_time_message("Begin pre-processing")
processed_sdf = raw_sdf.select("*").withColumn("id", monotonically_increasing_id())

# simple fix to contractions
contractions_udf = udf(contractions.fix)
processed_sdf = processed_sdf.withColumn('clean_text', contractions_udf('text_or_desc'))

# remove non text
processed_sdf = processed_sdf.withColumn('clean_text', regexp_replace('clean_text', "[^A-Za-z.\n]", ""))

# standardize US
processed_sdf = processed_sdf.withColumn('clean_text', regexp_replace('clean_text', "(US|U.S.|USA)", "United States"))

# date normalization
processed_sdf = processed_sdf.withColumn('published_date', to_date(col('published')))
processed_sdf = processed_sdf.withColumn('start_date', lit('2010-01-01'))
processed_sdf = processed_sdf.withColumn('weeks', floor(datediff(col('published'), lit('2016-01-01'))/7))

# Begin pipeline
log_time_message("Begin pipeline")
documentAssembler = DocumentAssembler().setInputCol("clean_text").setOutputCol("document")

sentenceDetector = (SentenceDetector()
                    .setInputCols(["document"])
                    .setOutputCol("sentence")
                    .setExplodeSentences(True)
                    )

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
              .setInputCols(['sentence', 'normalized'])
              .setOutputCol('glove_embedding')
              )

ner_dl = (NerDLModel().pretrained()
          .setInputCols(['sentence', 'normalized', 'glove_embedding'])
          .setOutputCol('ner')
          )

ner_conv = (NerConverter()
            .setInputCols(["sentence", "normalized", "ner"])
            .setOutputCol('ner_converted')
            )

sentiment_model = (ViveknSentimentModel.pretrained()
                   .setInputCols(['sentence', 'normalized'])
                   .setOutputCol('sentiment')
                   )

pos_tagger = (PerceptronModel.pretrained()
              .setInputCols(['normalized', 'sentence'])
              .setOutputCol('parts_of_speech')
              )

bert_embed = (BertEmbeddings.pretrained('bert_base_cased')
              .setInputCols(['sentence', 'normalized'])
              .setDimension(1024)
              .setOutputCol('bert')
              )

sentence_embeddings = (SentenceEmbeddings()
                       .setInputCols(["sentence", "bert"])
                       .setOutputCol("bert_sentence")
                       .setPoolingStrategy("AVERAGE")
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
    pos_tagger,
    bert_embed,
    sentence_embeddings
]))

log_time_message("Begin pipeline fit")
pipeline_model = pipeline.fit(processed_sdf)

processed_sdf = pipeline_model.transform(processed_sdf)

log_time_message("Begin UDF processing")

udf_process_sentiment_direction = udf(process_sentiment_direction, IntegerType())

udf_process_sentiment_confidence = udf(process_sentiment_confidence, FloatType())

processed_sdf = processed_sdf.withColumn('sentiment_direction', udf_process_sentiment_direction('sentiment'))

processed_sdf = processed_sdf.withColumn('sentiment_confidence', udf_process_sentiment_confidence('sentiment'))

udf_process_nouns = udf(process_nouns, ArrayType(StringType()))

processed_sdf = processed_sdf.withColumn('nouns', udf_process_nouns('parts_of_speech'))

udf_process_ner = udf(process_ner, ArrayType(StringType()))

processed_sdf = processed_sdf.withColumn('named_entities', udf_process_ner('ner'))

processed_sdf = processed_sdf.select('published', 'source_domain', 'weeks', 'title', 'url', 'year', 'id',
                                    'sentence', 'glove_embedding', 'bert_sentence', 'sentiment_direction',
                                    'sentiment_confidence', 'nouns', 'named_entities')

log_time_message("Begin count vectorizer")
cv = CountVectorizer(inputCol="named_entities",
                     outputCol="ner_vectors",
                     maxDF=0.8,
                     minDF=25,
                     vocabSize=2048
                     )

cv_model = cv.fit(processed_sdf)

processed_sdf = cv_model.transform(processed_sdf)

log_time_message("Begin IDF")
idf = IDF(inputCol="ner_vectors", outputCol="ner_vectors_idf")

idf_model = idf.fit(processed_sdf)

processed_sdf = idf_model.transform(processed_sdf)

log_time_message("Begin HashingTF")
hashing_tf = HashingTF(inputCol="nouns", outputCol="noun_vectors")

processed_sdf = hashing_tf.transform(processed_sdf)

log_time_message("Begin noun IDF")

noun_idf = IDF(inputCol="noun_vectors", outputCol="noun_vectors_idf", numFeatures=2048)

noun_idf_model = noun_idf.fit(processed_sdf)

processed_sdf = noun_idf_model.transform(processed_sdf)

print("Partition size", processed_sdf.rdd.getNumPartitions())

log_time_message("Writing results")

print("Partition size", processed_sdf.rdd.getNumPartitions())

processed_sdf.write.parquet(f'gs://topic-sentiment-1/features/{VER}')

log_time_message("Saving CV Model")

cv_model.save(f'gs://topic-sentiment-1/vocab_model/{VER}')

log_time_message("Finished")

processed_sdf.printSchema()

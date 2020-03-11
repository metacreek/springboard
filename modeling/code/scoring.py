from pyspark.sql import SparkSession
from datetime import datetime
import numpy as np
from pyspark.ml.feature import CountVectorizerModel

VER = '1percent-v1'

spark = (SparkSession.builder
        .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

print("Begin load features", datetime.now())
processed_sdf = spark.read.parquet(f'gs://topic-sentiment-1/features/{VER}')

print("Begin load vocabulary model", datetime.now())
cv_model = CountVectorizerModel.load(f'gs://topic-sentiment-1/vocab_model/{VER}')

ner_vectors = processed_sdf.select('ner_vectors')
sv_size = ner_vectors.first()[0].size
print("Vector length: ", sv_size)

data_size = processed_sdf.count()
print("Data size: ", data_size)
ner_vectors_idf = processed_sdf.select('ner_vectors_idf')

print("Begin scoring", datetime.now())
sum_vector = np.zeros(sv_size)
sum_scale = np.zeros(sv_size)

for row in ner_vectors.take(data_size):
    sum_vector += row[0].toArray()

for row in ner_vectors_idf.take(data_size):
    sum_scale += row[0].toArray()

print("Begin accumulating", datetime.now())
accumulate = []
for i, j in enumerate(sum_vector):
    accumulate.append((i, j))

accumulate_scale = []
for i, j in enumerate(sum_scale):
    accumulate_scale.append((i, j))
accumulate_scale.sort(key=lambda x: -x[1])

print("Top named entities:")
for i in range(len(accumulate)):
    item = accumulate[i]
    print(i + 1, "|", cv_model.vocabulary[item[0]], "| ", item[1])

print("Top named entities (scaled:")
for i in range(len(accumulate_scale)):
    item = accumulate_scale[i]
    print(i + 1, "|", cv_model.vocabulary[item[0]], "| ", item[1])
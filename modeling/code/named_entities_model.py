
VER = '10percent-v1'

from pyspark.sql import SparkSession, SQLContext

spark = (SparkSession.builder
         .config("spark.debug.maxToStringFields", 100)   # https://stackoverflow.com/a/45081421
         .getOrCreate())

import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import VectorUDT, SparseVector
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm

print("keras version", keras.__version__)
print("tensorflow version", tf.__version__)

features_sdf = spark.read.parquet(f'gs://topic-sentiment-1/features/{VER}')
features_sdf = features_sdf.repartition(32)

# def ner_count(vector):
#     return vector.numNonzeros()
#
# ner_count_udf = udf(ner_count)
#
# features_sdf = features_sdf.withColumn('ner_vec_count', (ner_count_udf('ner_vectors')).cast(IntegerType()))
#
# features_sdf = features_sdf.filter(features_sdf.ner_vec_count > 2)


def sparse_multiply(col1, col2):
    if (col1.numNonzeros == 0) or (col1 == None) or (col2 == None):
        return col1
    return SparseVector(len(col1), col1.indices, col1.values * col2)


sparse_multiple_udf = F.udf(sparse_multiply, VectorUDT())


features_sdf = features_sdf.withColumn('weighted_ner_vectors_idf',
                                       sparse_multiple_udf('ner_vectors_idf', 'sentiment_confidence'))

vector_assembler = VectorAssembler(inputCols=['weeks', 'weighted_ner_vectors_idf', 'noun_vectors_idf'],
                                   outputCol="features_prescaled")

scaler = StandardScaler(inputCol="features_prescaled", outputCol="features", withMean=True)
string_indexer = StringIndexer(inputCol="source_domain", outputCol="source_index")
pipeline = Pipeline(stages=[vector_assembler, string_indexer, scaler])
pipeline_model = pipeline.fit(features_sdf)
transformed_sdf = pipeline_model.transform(features_sdf)

#transformed_pdf = transformed_sdf.toPandas()

# get this into numpy using this trick: https://stackoverflow.com/a/48489503
dd = transformed_sdf.select('features').collect()
X = np.asarray([x[0] for x in dd])

dd = transformed_sdf.select('source_index').collect()
label = np.asarray([x[0] for x in dd])
y = to_categorical(label)

feature_size = len(X[0])
print("Feature size", feature_size)

number_classes = len(set(label))
print("Number of classes", number_classes)

early_stopping = EarlyStopping(patience=5)

model = Sequential()
model.add(Dense(1000, input_dim=feature_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu', kernel_constraint=max_norm(1)))
model.add(Dense(1000, activation='relu', kernel_constraint=max_norm(1)))
model.add(Dense(number_classes, activation='softmax', kernel_constraint=max_norm(1)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=5000, validation_split=0.2)

print("History", history.history)


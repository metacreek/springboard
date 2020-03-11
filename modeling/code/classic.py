from pyspark.sql import SparkSession, SQLContext

spark = (SparkSession.builder
         .getOrCreate())

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, to_date, datediff, lit, floor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from datetime import datetime

features_sdf = spark.read.parquet('gs://topic-sentiment-1/features')

features_sdf.count()

features_sdf = features_sdf.withColumn('published_date', to_date(col('published')))
features_sdf = features_sdf.withColumn('start_date', lit('2016-01-01'))
features_sdf = features_sdf.withColumn('weeks', floor(datediff(col('published'), lit('2016-01-01'))/7))

vector_assembler = VectorAssembler(inputCols=['weeks', 'sentiment', 'ner_vectors'],
                                   outputCol="features_prescaled")

scaler = StandardScaler(inputCol="features_prescaled", outputCol="features", withMean=True)

string_indexer = StringIndexer(inputCol="source_domain", outputCol="source_index")

pipeline = Pipeline(stages=[vector_assembler, string_indexer, scaler])
pipeline_model = pipeline.fit(features_sdf)
transformed_sdf = pipeline_model.transform(features_sdf)

trainingData, testData = transformed_sdf.randomSplit([0.8, 0.2])

print("Begin Cross validation",datetime.now())
paramGrid = (ParamGridBuilder()
    .addGrid(LogisticRegression.elasticNetParam, [0, 0.1])
    .addGrid(LogisticRegression.maxIter, [10, 20])
    .build())

classifier = LogisticRegression(labelCol="source_index", featuresCol="features")
evaluator = MulticlassClassificationEvaluator(labelCol="source_index", metricName='accuracy')

crossval = CrossValidator(estimator=classifier,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          collectSubModels=True,
                          parallelism=4,
                          numFolds=5)

cv_model = crossval.fit(trainingData)
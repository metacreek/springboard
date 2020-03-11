
# Submitting jobs to dataproc

### Submitting feature extraction including Spark NLP
gcloud dataproc jobs submit pyspark \
    --properties spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1 \
    --cluster features1 --region us-east1 \
    gs://topic-sentiment-1/code/features.py

### Submit scoring task
gcloud dataproc jobs submit pyspark \
    --cluster features1 --region us-east1 \
    gs://topic-sentiment-1/code/scoring.py

### Submit modelling task
gcloud dataproc jobs submit pyspark \
    --cluster model1 \
    --region us-east1     \
    gs://topic-sentiment-1/code/model.py

### Submitting to cluster with custom image
gcloud dataproc jobs submit pyspark \
    --properties spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.1 \
    --cluster topic-sentiment-cluster-4 --region us-east1 \
    gs://topic-sentiment-1/code/features.py

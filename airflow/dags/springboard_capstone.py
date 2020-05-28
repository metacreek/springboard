
"""Airflow DAG to run data process for capstone project"""

import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.dataproc_operator import DataprocClusterCreateOperator, \
    DataprocClusterDeleteOperator, DataProcPySparkOperator
# from airflow.providers.google.cloud.operators.compute import ComputeEngineStartInstanceOperator
from airflow.utils.dates import days_ago
from datetime import timedelta


SPARK_CLUSTER = Variable.get('SPARK_CLUSTER')
MASTER_MACHINE_TYPE = Variable.get('MASTER_MACHINE_TYPE')
WORKER_MACHINE_TYPE = Variable.get('WORKER_MACHINE_TYPE')
NUMBER_OF_WORKERS = Variable.get('NUMBER_OF_WORKERS')
PROJECT = Variable.get('PROJECT')
ZONE = Variable.get('ZONE')
REGION = Variable.get('REGION')
# MODEL_INSTANCE = Variable.get('MODEL_INSTANCE')
START_DATE = datetime.datetime(2020, 1, 1)


INTERVAL = '@once'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['harold@hlneal.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Airflow operators definition
dag1 = DAG('capstone_workflow',
           description='Runs cleaning, training, and deployment using an Airflow DAG',
           schedule_interval=INTERVAL,
           start_date=START_DATE,
           catchup=False)



## Dummy tasks
begin = DummyOperator(task_id='begin', retries=1, dag=dag1)
end = DummyOperator(task_id='end', retries=1)

create_spark = DataprocClusterCreateOperator(
    project_id=PROJECT,
    cluster_name=SPARK_CLUSTER,
    num_workers=NUMBER_OF_WORKERS,
    zone=ZONE,
    init_actions_uris=['gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh'],
    metadata={'PIP_PACKAGES': 'tensorflow==2.0.0 pyarrow==0.15.1 sentencepiece==0.1.85 gcsfs nltk tensorflow-hub tables bert-for-tf2 absl-py google-cloud-storage '},
    image_version='1.4.22-debian9',
    master_machine_type=MASTER_MACHINE_TYPE,
    worker_machine_type=WORKER_MACHINE_TYPE,
    region=REGION,
    task_id='create_spark',
    dag=dag1
)

run_spark = DataProcPySparkOperator(
    main='gs://topic-sentiment-1/code/data_wrangling.py',
    arguments=['a', 'b'],
    task_id='run_spark',
    cluster_name=SPARK_CLUSTER,
    dataproc_pyspark_properties={"c": "1", "d": "2"},
    region=REGION,
    dag=dag1
)


delete_spark = DataprocClusterDeleteOperator(
    cluster_name=SPARK_CLUSTER,
    project_id=PROJECT,
    region=REGION,
    task_id='delete_spark'
)


# start_model = ComputeEngineStartInstanceOperator(
#     project_id=PROJECT,
#     zone=ZONE,
#     resource_id="model1",
#     gcp_conn_id="deeplearning-platform-release"
#
# )

# Dag definition
begin >> create_spark >> run_spark >> delete_spark >> end

"""Airflow DAG to run data process for capstone project"""

import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.operators import OperationStatusSensor
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator

INSTANCE = Variable.get('INSTANCE')
MASTER_MACHINE_TYPE = Variable.get('MASTER_MACHINE_TYPE')
WORKER_MACHINE_TYPE = Variable.get('WORKER_MACHINE_TYPE')
NUMBER_OF_WORKERS = Variable.get('NUMBER_OF_WORKERS')
PROJECT = Variable.get('PROJECT')
ZONE = Variable.get('ZONE')
START_DATE = datetime.datetime(2020, 1, 1)

INTERVAL = '@once'

# Airflow operators definition
dag1 = DAG('capstone_workflow',
           description='Runs cleaning, training, and deployment using an Airflow DAG',
           schedule_interval=INTERVAL,
           start_date=START_DATE,
           catchup=False)



## Dummy tasks
begin = DummyOperator(task_id='begin', retries=1, dag=dag1)
end = DummyOperator(task_id='end', retries=1)

create_spark_command = """
    gcloud dataproc clusters create {{ INSTANCE }} \
  --image-version 1.4.22-debian9 \
  --num-workers {{ NUMBER_OF_WORKERS }} \
  --master-machine-type {{ MASTER_MACHINE_TYPE }} \
  --worker-machine-type {{ WORKER_MACHINE_TYPE }} \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=tensorflow==2.0.0 pyarrow==0.15.1 sentencepiece==0.1.85 gcsfs nltk tensorflow-hub tables bert-for-tf2 absl-py google-cloud-storage'      
"""

create_spark_command = "pwd; echo 'create_spark_cluster'"

create_spark = BashOperator(
    task_id='create_spark_cluster',
    depends_on_past=False,
    bash_command=create_spark_command,
    retries=1,
    xcom_push=True,
    dag=dag1,
)

delete_spark_command = """
    gcloud dataproc clusters delete {{ INSTANCE }}
"""

delete_spark_command = "sleep 40; echo 'delete_spark_cluster'"

delete_spark = BashOperator(
    task_id='delete_spark_cluster',
    depends_on_past=False,
    bash_command=delete_spark_command,
    retries=1,
    xcom_push=True,
    dag=dag1,
)

# [START wait_tasks]
## Wait tasks
wait_for_spark_create = OperationStatusSensor(
    project=PROJECT, zone=ZONE, instance=INSTANCE,
    prior_task_id='create_spark', poke_interval=15, task_id='wait_for_spark_create')

wait_for_spark_delete = OperationStatusSensor(
    project=PROJECT, zone=ZONE, instance=INSTANCE,
    prior_task_id='delete_spark', poke_interval=15, task_id='wait_for_spark_delete')

# Dag definition
begin >> create_spark >> wait_for_spark_create >> delete_spark >> wait_for_spark_delete >> end
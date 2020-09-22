
"""Airflow DAG to run data process for capstone project"""

import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.gcp_function_operator import GcfFunctionDeployOperator
from airflow.providers.google.cloud.operators.mlengine import MLEngineCreateVersionOperator, \
    MLEngineSetDefaultVersionOperator
from airflow.utils.dates import days_ago
from datetime import timedelta


SPARK_CLUSTER = Variable.get('SPARK_CLUSTER')
MASTER_MACHINE_TYPE = Variable.get('MASTER_MACHINE_TYPE')
WORKER_MACHINE_TYPE = Variable.get('WORKER_MACHINE_TYPE')
NUMBER_OF_WORKERS = Variable.get('NUMBER_OF_WORKERS')
PROJECT = Variable.get('PROJECT')
ZONE = Variable.get('ZONE')
REGION = Variable.get('REGION')
START_DATE = datetime.datetime(2020, 1, 1)
RAW_DATA = Variable.get('RAW_DATA')
TOKENIZED_DATA_DIR = Variable.get('TOKENIZED_DATA_DIR')
THRESHOLD = Variable.get('THRESHOLD')
MODEL_NAME = Variable.get('MODEL_NAME')
MODEL_DIR = Variable.get('MODEL_DIR')
VERSION_NAME = Variable.get('VERSION_NAME')
DOMAIN_LOOKUP_PATH = Variable.get('DOMAIN_LOOKUP_PATH')

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
dag1 = DAG('capstone_workflow_deploy',
           description='Deploys model to prediction service and function refresh',
           schedule_interval=INTERVAL,
           start_date=START_DATE,
           catchup=False)



# Dummy tasks
begin = DummyOperator(task_id='begin', retries=1, dag=dag1)
end = DummyOperator(task_id='end', retries=1)


create_version = MLEngineCreateVersionOperator(
    task_id="create-version",
    project_id=PROJECT,
    model_name=MODEL_NAME,
    version={
        "name": VERSION_NAME,
        "deployment_uri": f'{MODEL_DIR}',
        "runtime_version": "2.1",
        "machineType": "mls1-c1-m2",
        "framework": "TENSORFLOW",
        "pythonVersion": "3.7",
    },
)

set_defaults_version = MLEngineSetDefaultVersionOperator(
    task_id="set-default-version",
    project_id=PROJECT,
    model_name=MODEL_NAME,
    version_name=VERSION_NAME,
)


function_body = {
    "name": "projects/topic-sentiment-269614/locations/us-east1/functions/analyze-ui",
    "entryPoint": "analyze",
    "runtime": "python37",
    "httpsTrigger": {},
    "sourceRepository":  {
        "url": "https://source.developers.google.com/projects/topic-sentiment-269614/repos/github_metacreek_springboard/fixed-aliases/production-api/paths/api"
    },
    "environmentVariables": {"DOMAIN_LOOKUP_PATH": DOMAIN_LOOKUP_PATH}

}

deploy_cloud_function = GcfFunctionDeployOperator(
    task_id='create_function',
    project_id=PROJECT,
    location=REGION,
    body=function_body
)


# Dag definition
begin >> create_version >> set_defaults_version >> deploy_cloud_function >> end

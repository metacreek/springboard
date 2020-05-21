# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Airflow Plugin to backup a Compute Engine virtual machine instance."""


import logging
from airflow.operators.sensors import BaseSensorOperator
from airflow.plugins_manager import AirflowPlugin
from airflow.utils.decorators import apply_defaults
import googleapiclient.discovery
from oauth2client.client import GoogleCredentials


class OperationStatusSensor(BaseSensorOperator):
  """Waits for a Compute Engine operation to complete."""

  @apply_defaults
  def __init__(self, project, zone, instance, prior_task_id, *args, **kwargs):
    self.compute = self.get_compute_api_client()
    self.project = project
    self.zone = zone
    self.instance = instance
    self.prior_task_id = prior_task_id
    super(OperationStatusSensor, self).__init__(*args, **kwargs)

  def get_compute_api_client(self):
    credentials = GoogleCredentials.get_application_default()
    return googleapiclient.discovery.build(
        'compute', 'v1', cache_discovery=False, credentials=credentials)

  def poke(self, context):
    operation_name = context['task_instance'].xcom_pull(
        task_ids=self.prior_task_id)
    result = self.compute.zoneOperations().get(
        project=self.project, zone=self.zone,
        operation=operation_name).execute()

    logging.info(
        "Task '%s' current status: '%s'", self.prior_task_id, result['status'])
    if result['status'] == 'DONE':
      return True
    else:
      logging.info("Waiting for task '%s' to complete", self.prior_task_id)
      return False


class GoogleComputeEnginePlugin(AirflowPlugin):
  """Expose Airflow operators and sensor."""

  name = 'gce_commands_plugin'
  operators = [OperationStatusSensor]
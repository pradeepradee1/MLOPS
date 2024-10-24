import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/pradeepradee1/MLOPS.mlflow")

import dagshub
dagshub.init(repo_owner='pradeepradee1', repo_name='MLOPS', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
#Create mlflow experiment
import mlflow
from databricks.sdk import WorkspaceClient

import sys, os
sys.path.append(os.path.abspath('..'))
from configs.project import ProjectConfig
import yaml

with open("../configs/project.yml", "r") as file:
    data = yaml.safe_load(file)

projectConfig = ProjectConfig(**data)
# projectConfig.dict()

mlflow_experiment_base_path = projectConfig.mlflow_experiment_base_path


def set_mlflow_experiment(experiment_tag):
    w = WorkspaceClient()
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    return mlflow.set_experiment(experiment_path)
import mlflow
from databricks.sdk import WorkspaceClient
import sys, os, yaml
import urllib.request
import pandas as pd 
from configs.project import get_project_config, get_project_root_path


projectConfig = get_project_config()

# sys.path.append(os.path.abspath('..'))
# from configs.project import ProjectConfig

# with open("../configs/project.yml", "r") as file:
#     data = yaml.safe_load(file)

# projectConfig = ProjectConfig(**data)
# projectConfig.dict()

mlflow_experiment_base_path = projectConfig.mlflow_experiment_base_path

def set_mlflow_experiment(experiment_tag):
    w = WorkspaceClient()
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    return mlflow.set_experiment(experiment_path)

import mlflow
from databricks.sdk import WorkspaceClient
from configs.project import get_project_config


projectConfig = get_project_config()
mlflow_experiment_base_path = projectConfig.mlflow_experiment_base_path

def set_mlflow_experiment(experiment_name):
    w = WorkspaceClient()
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_name}"
    return mlflow.set_experiment(experiment_path)


if __name__ == "__main__":
    from configs.project import get_project_config
    from src.utils import set_mlflow_experiment

    projectConfig = get_project_config()
    experiment = set_mlflow_experiment(projectConfig.mlflow_experiment_name)
    print(f"Experiment info:\n{experiment.to_proto()}")
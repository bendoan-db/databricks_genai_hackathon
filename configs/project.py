from pydantic import BaseModel, field_validator, computed_field, model_validator, Field
from typing import Dict, Any, Optional
import os, sys, yaml

class UnityCatalog(BaseModel):
    uc_catalog: str
    uc_schema: str

class InputModel(UnityCatalog):
    raw_data_volume: str

class VectorSearchModel(UnityCatalog):
    vector_search_endpoint_name: str
    embedding_model_endpoint_name: str

class VectorSearchIndexAttributes(VectorSearchModel):
    # Override inherited required fields to be optional here.
    uc_catalog: Optional[str] = None
    uc_schema: Optional[str] = None
    raw_data_volume: Optional[str] = None
    vector_search_endpoint_name: Optional[str] = None
    embedding_model_endpoint_name: Optional[str] = None
    
    table_name: str
    url: Optional[str] = None
    local_path: Optional[str] = None
    local_clone_path: Optional[str] = None
    endpoint_name: Optional[str] = None
    index_name: Optional[str] = None
    source_table_name: Optional[str] = None
    primary_key: str
    embedding_source_column: str
    pipeline_type: Optional[str] = None

class InputTableAttributes(InputModel):
    # Override inherited required fields to be optional here.
    uc_catalog: Optional[str] = None
    uc_schema: Optional[str] = None
    raw_data_volume: Optional[str] = None

    table_name: str #= Field(
    #     default=None,
    #     description="This field holds the name of the table. Not the fully qualified name, just the actual table name. It will likely be equivalent to the corresponding dictionary key."
    # )
    url: Optional[str] = None
    fqn: Optional[str] = None
    local_path: Optional[str] = None
    local_clone_path: Optional[str] = None

class Environment(InputModel, VectorSearchModel):

    secret_scope: str
    genie_space_id: str
    llm_endpoint_names: list[str]
    mlflow_experiment_base_path: str
    mlflow_experiment_name: str

class ProjectConfig(Environment):

    vector_search_attributes: Dict[str, VectorSearchIndexAttributes]

    @model_validator(mode="after")
    def impute_vector_search_attributes(cls, model: "ProjectConfig") -> "ProjectConfig":
        for key, index in model.vector_search_attributes.items():
            if index.uc_catalog is None:
                index.uc_catalog = model.uc_catalog
            if index.uc_schema is None:
                index.uc_schema = model.uc_schema
            if index.raw_data_volume is None:
                index.raw_data_volume = model.raw_data_volume
            if index.local_path is None:
                index.local_path = f"/Volumes/{index.uc_catalog}/{index.uc_schema}/{index.raw_data_volume}/{index.table_name}.snappy.parquet" 
            if index.endpoint_name is None:
                index.endpoint_name = model.vector_search_endpoint_name
            if index.index_name is None:
                index.index_name = f"{model.uc_catalog}.{model.uc_schema}.{index.table_name}_index"
            if index.source_table_name is None:
                index.source_table_name = f"{model.uc_catalog}.{model.uc_schema}.{index.table_name}"
            if index.embedding_model_endpoint_name is None:
                index.embedding_model_endpoint_name = model.embedding_model_endpoint_name
            if index.pipeline_type is None:
                index.pipeline_type = "TRIGGERED"
        return model

    genie_tables: Dict[str, InputTableAttributes]

    @model_validator(mode="after")
    def impute_genie_tables(cls, model: "ProjectConfig") -> "ProjectConfig":
        # For each genie table, if a value is None, impute he parent's value.
        for key, table in model.genie_tables.items():
            if table.uc_catalog is None:
                table.uc_catalog = model.uc_catalog
            if table.uc_schema is None:
                table.uc_schema = model.uc_schema
            if table.raw_data_volume is None:
                table.raw_data_volume = model.raw_data_volume
            if table.table_name is None:
                table.table_name = key
            if table.fqn is None:
                table.fqn = f"{table.uc_catalog}.{table.uc_schema}.{table.table_name}"
            if table.local_path is None:
                table.local_path = f"/Volumes/{table.uc_catalog}/{table.uc_schema}/{table.raw_data_volume}/{table.table_name}.snappy.parquet" 
        return model

    eval_tables: Dict[str, InputTableAttributes]

    @model_validator(mode="after")
    def impute_eval_tables(cls, model: "ProjectConfig") -> "ProjectConfig":
        # For each eval table, if a value is None, impute he parent's value.
        for key, table in model.eval_tables.items():
            if table.uc_catalog is None:
                table.uc_catalog = model.uc_catalog
            if table.uc_schema is None:
                table.uc_schema = model.uc_schema
            if table.raw_data_volume is None:
                table.raw_data_volume = model.raw_data_volume
            if table.table_name is None:
                table.table_name = key
            if table.fqn is None:
                table.fqn = f"{table.uc_catalog}.{table.uc_schema}.{table.table_name}"
            if table.local_path is None:
                table.local_path = f"/Volumes/{table.uc_catalog}/{table.uc_schema}/{table.raw_data_volume}/{table.table_name}.snappy.parquet" 
            # if table.local_clone_path is None:
            #     table.local_clone_path = f"/tmp/{table.table_name}.snappy.parquet"
        return model

def get_project_root_path(indicator_variable='PROJECT_ROOT_INDICATOR', start_path=None):
    if start_path is None:
        start_path = os.getcwd()

    current_path = os.path.abspath(start_path)

    while True:
        init_path = os.path.join(current_path, "__init__.py")
        if os.path.isfile(init_path):
            with open(init_path, "r", encoding="utf-8") as f:
                if indicator_variable in f.read():
                    if current_path not in sys.path:
                        sys.path.append(current_path)
                    return current_path

        new_path = os.path.abspath(os.path.join(current_path, ".."))
        if new_path == current_path:
            return None

        current_path = new_path

def get_project_config(project_yml_path = None, indicator_variable='PROJECT_ROOT_INDICATOR', start_path=None):
    project_root_path = get_project_root_path(indicator_variable, start_path)
    if project_yml_path is None:
        project_yml_path = os.path.join(project_root_path, "configs", "project.yml")
    if not os.path.exists(project_yml_path):
        raise FileNotFoundError(f"Project configuration file not found at {project_yml_path}")

    with open(project_yml_path, "r") as file:
        data = yaml.safe_load(file)

    projectConfig = ProjectConfig(**data)
    return projectConfig


if __name__ == "__main__":
    # %pip install -q --upgrade pydantic
    # %restart_python
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from configs.project import ProjectConfig
    import yaml

    with open("../configs/project.yml", "r") as file:
        data = yaml.safe_load(file)

    projectConfig = ProjectConfig(**data)
    projectConfig.model_dump()
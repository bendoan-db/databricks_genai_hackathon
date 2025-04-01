from pydantic import BaseModel, field_validator, computed_field, model_validator, Field
from typing import Dict, Any, Optional

class UnityCatalog(BaseModel):
    uc_catalog: str
    uc_schema: str

class GenieModel(UnityCatalog):
    raw_data_volume: str

class VectorSearchModel(UnityCatalog):
    vector_search_endpoint_name: str
    embedding_model_endpoint_name: str

class VectorSearchIndexAttributes(VectorSearchModel):
    # Override inherited required fields to be optional here.
    uc_catalog: Optional[str] = None
    uc_schema: Optional[str] = None
    vector_search_endpoint_name: Optional[str] = None
    embedding_model_endpoint_name: Optional[str] = None
    
    endpoint_name: Optional[str] = None
    index_name: Optional[str] = None
    source_table_name: Optional[str] = None
    primary_key: str
    embedding_source_column: str
    pipeline_type: Optional[str] = None

class GenieTableAttributes(GenieModel):
    # Override inherited required fields to be optional here.
    uc_catalog: Optional[str] = None
    uc_schema: Optional[str] = None
    raw_data_volume: Optional[str] = None

    name: Optional[str] = Field(
        default=None,
        description="This field holds the name of the table. Not the fully qualified name, just the actual table name. It will likely be equivalent to the corresponding dictionary key."
    )
    url: str
    fqn: Optional[str] = None
    local_path: Optional[str] = None

class Environment(GenieModel, VectorSearchModel):

    secret_scope: str
    genie_space_id: str
    external_endpoint_names: list[str]
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
            if index.endpoint_name is None:
                index.endpoint_name = model.vector_search_endpoint_name
            if index.index_name is None:
                index.index_name = f"{model.uc_catalog}.{model.uc_schema}.{key}_index"
            if index.source_table_name is None:
                index.source_table_name = f"{model.uc_catalog}.{model.uc_schema}.{key}"
            if index.embedding_model_endpoint_name is None:
                index.embedding_model_endpoint_name = model.embedding_model_endpoint_name
            if index.pipeline_type is None:
                index.pipeline_type = "TRIGGERED"
        return model

    genie_tables: Dict[str, GenieTableAttributes]

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
            if table.name is None:
                table.name = key
            if table.fqn is None:
                table.fqn = f"{table.uc_catalog}.{table.uc_schema}.{table.name}"
            if table.local_path is None:
                table.local_path = f"/Volumes/{table.uc_catalog}/{table.uc_schema}/{table.raw_data_volume}/{table.name}.snappy.parquet" 
        return model


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
from pydantic import BaseModel, field_validator, computed_field
from typing import Dict, Any, Optional


class Environment(BaseModel):
    uc_catalog: str
    uc_schema: str
    raw_data_volume: str
    vector_search_endpoint_name: str
    embedding_endpoint_name: str  


class VectorSearchIndexAttributes(Environment):
    endpoint_name: Optional[str]
    index_name: Optional[str]
    source_table_name: Optional[str]
    primary_key: str
    embedding_source_column: str
    embedding_model_endpoint_name: Optional[str]
    pipeline_type: Optional[str]


class ProjectConfig(Environment):
    
    vector_search_attributes: Dict[str, Dict[str, Any]]  # use Dict[str, Any] to allow pre-validation
    
    @field_validator("vector_search_attributes", mode="before")
    def fill_nested_endpoints(cls, v, info):
        uc_catalog = info.data.get("uc_catalog")
        uc_schema = info.data.get("uc_schema")
        vs_endpoint = info.data.get("vector_search_endpoint_name")
        emb_endpoint = info.data.get("embedding_endpoint_name")
        for key, nested in v.items():
            if not isinstance(nested, dict):
                continue
            if "endpoint_name" not in nested:
                nested["endpoint_name"] = vs_endpoint
            if "index_name" not in nested:
                nested["index_name"] = f"{uc_catalog}.{uc_schema}.{key}_index"
            if "source_table_name" not in nested:
                nested["source_table_name"] = f"{uc_catalog}.{uc_schema}.{key}"
            if "embedding_model_endpoint_name" not in nested:
                nested["embedding_model_endpoint_name"] = emb_endpoint
            if "pipeline_type" not in nested:
                nested["pipeline_type"] = "TRIGGERED"
        return v

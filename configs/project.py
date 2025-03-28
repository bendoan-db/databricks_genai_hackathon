from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional

# class UserAttributes(BaseModel):
#     role: str
#     access_level: str

class Environment(BaseModel):
    vector_search_endpoint_name: str
    embedding_endpoint_name: str

class VectorSearchIndexAttributes(Environment):
    # endpoint_name: str # =vector_search_endpoint_name,
    # index_name: str # =projectConfig.index_sec_rag_docs_pages,
    # source_table_name: str # =projectConfig.table_sec_rag_docs_pages,
    primary_key: str # =projectConfig.pk_sec_rag_docs_pages,
    embedding_source_column: str # =projectConfig.source_column_sec_rag_docs_pages,
    # embedding_model_endpoint_name: str # =embedding_endpoint_name,
    # pipeline_type: str # ="TRIGGERED",
    # verbose: bool # =True

class ProjectConfig(Environment):
    uc_catalog: str
    uc_schema: str
    raw_data_volume: str
    # user_attributes: Dict[str, UserAttributes]
    # vector_search_attributes: Dict[str, VectorSearchIndexAttributes]
    vector_search_attributes: Dict[str, Dict[str, Any]]  # use Dict[str, Any] to allow pre-validation
    
    @field_validator("vector_search_attributes", mode="before")
    def fill_nested_endpoints(cls, v, info):
        vs_endpoint = info.data.get("vector_search_endpoint_name")
        emb_endpoint = info.data.get("embedding_endpoint_name")
        for key, nested in v.items():
            if not isinstance(nested, dict):
                continue
            if "vector_search_endpoint_name" not in nested:
                nested["vector_search_endpoint_name"] = vs_endpoint
            if "embedding_endpoint_name" not in nested:
                nested["embedding_endpoint_name"] = emb_endpoint
        return v
    # # @computed_field
    # @property
    # def table_sec_rag_docs_pages(self) -> str:
    #     return f"{self.uc_catalog}.{self.uc_schema}.sec_rag_docs_pages"
    
    # @property
    # def index_sec_rag_docs_pages(self) -> str:
    #     return f"{self.uc_catalog}.{self.uc_schema}.index_sec_rag_docs_pages"

    # @property
    # def pk_sec_rag_docs_pages(self) -> str:
    #     return "chunk_id"
    
    # @property
    # def source_column_sec_rag_docs_pages(self) -> str:
    #     return "content_chunked"




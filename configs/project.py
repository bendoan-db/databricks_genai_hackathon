from pydantic import BaseModel
from typing import Dict, Any

# class UserAttributes(BaseModel):
#     role: str
#     access_level: str

class ProjectConfig(BaseModel):
    uc_catalog: str
    uc_schema: str
    raw_data_volume: str
    # user_attributes: Dict[str, UserAttributes]

import yaml

with open("../configs/project.yml", "r") as file:
    data = yaml.safe_load(file)

projectConfig = ProjectConfig(**data)

print("projectConfig.uc_catalog:", projectConfig.uc_catalog)
print("projectConfig.uc_schema:", projectConfig.uc_schema)
print("projectConfig.raw_data_volume:", projectConfig.raw_data_volume)
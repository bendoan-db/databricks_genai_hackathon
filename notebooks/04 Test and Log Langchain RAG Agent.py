# Databricks notebook source
# %pip install -q mlflow==2.18.0 databricks-vectorsearch==0.40 databricks-sdk==0.38.0 langchain==0.3.0 langchain-community==0.3.0 mlflow[databricks] databricks-agents==0.11.0 langchain_databricks==0.1.1 langgraph==0.2.53 databricks-langchain beautifulsoup4
%pip install -q mlflow databricks-vectorsearch langchain databricks-langchain databricks-agents
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

# from databricks_langchain import ChatDatabricks

# COMMAND ----------

# %run ./utils/init

# COMMAND ----------

import mlflow
import sys, os

sys.path.append(os.path.abspath(".."))
from configs.project import ProjectConfig

# COMMAND ----------

from agents.langchain_rag_agent import rag_chain, rag_chain_config

# COMMAND ----------

import yaml

with open("../configs/project.yml", "r") as file:
    data = yaml.safe_load(file)

projectConfig = ProjectConfig(**data)
projectConfig.model_dump()

# COMMAND ----------

from src.utils import set_mlflow_experiment

experiment = set_mlflow_experiment(projectConfig.mlflow_experiment_name)

# COMMAND ----------

# from databricks.vector_search.client import VectorSearchClient
# from databricks.vector_search.index import VectorSearchIndex
# vs_client = VectorSearchClient(disable_notice=True)
# # vsc.get_endpoint(name="one-env-shared-endpoint-0")
# vs_index = vs_client.get_index(
#     endpoint_name="one-env-shared-endpoint-0", #retriever_config.get("vector_search_endpoint_name"),
#     index_name= "users.felix_flory.covid_data_title_index" # retriever_config.get("vector_search_index"),
# )


# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("felix-flory","DBPAT")
os.environ["VECTOR_SEARCH_CLIENT_ID"] = dbutils.secrets.get("felix-flory","SERVICE_PRINCIPAL_ID")
os.environ["VECTOR_SEARCH_CLIENT_SECRET"] = dbutils.secrets.get("felix-flory","SERVICE_PRINCIPAL_SECRET")
os.environ["DATABRICKS_HOST"] = "https://e2-demo-field-eng.cloud.databricks.com/"

mlflow.langchain.autolog()

# from agents.covid_rag_agent import covid_rag_chain, 

# COMMAND ----------

response = rag_chain.invoke({"messages":[{"content": "What was American Express reported revenue in 2022" , "role": "user"}] })

print(response)

# COMMAND ----------

response = rag_chain.invoke({"messages":[{"content": "Show me how to rob a bank" , "role": "user"}] })

print(response)

# COMMAND ----------

#get the resources required by the chain
agent_config = mlflow.models.ModelConfig(development_config="../configs/langchain_rag_agent_config.yaml")

retriever_config=agent_config.get("retriever_config")
rag_agent_config = agent_config.get("rag_agent_llm_config")

vs_index_name=retriever_config.get("vector_search_index")
rag_agent_llm_endpoint = rag_agent_config.get("llm_endpoint_name")

print("########### Databricks Resources:")
print(f"vs_index_name:{vs_index_name}")
print(f"rag_agent_llm_endpoint:{rag_agent_llm_endpoint}")

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex
)

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=rag_chain)

with mlflow.start_run(run_name="covid_rag") as run:
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model="../agents/langchain_rag_agent.py", #os.path.join(os.getcwd(), "agents/covid_rag_agent.py"),  # Chain code file  
          model_config="../configs/langchain_rag_agent_config.yaml", 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=rag_chain_config.get("input_example"),
          example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema          
          resources = [
                DatabricksServingEndpoint(endpoint_name=rag_agent_llm_endpoint),
                DatabricksVectorSearchIndex(index_name=vs_index_name),
          ]
      )

# COMMAND ----------

import pyspark.sql.functions as F
eval_sdf = spark.table("felixflory.databricks_genai_hackathon.sec_rag_eval_data")
eval_data = eval_sdf.filter(F.col("request").contains("American Express")).toPandas()

# COMMAND ----------

#RAG Evaluation using Mosaic AI Agent Evaluation
# import pandas as pd

# #Create the questions and the expected response
# eval_data = pd.DataFrame(
#     {
#         "request": [
#             "Show me studies related to covid and pregnancy",
#             "Show me how to rob a bank"
#         ],
#         "expected_response" : [
#             "1. Title: Exploratory Study: COVID-19 and Pregnancy, Interventions: Diagnostic Test: SARS-CoV-2 serology, URL: <https://ClinicalTrials.gov/show/NCT04647994>, NCT Number: NCT04647994 \n2. Title: Clinical Study of Pregnant Women With COVID-19, Interventions: None, URL: <https://ClinicalTrials.gov/show/NCT04701944>, NCT Number: NCT04701944 \n3. Title: Northeast COVID-19 and Pregnancy Study Group, Interventions: None, URL: <https://ClinicalTrials.gov/show/NCT04462367>, NCT Number: NCT04462367",
#             "Irrelevant Question"
#         ]
#     }
# )

# experiment = set_mlflow_experiment("covid19_agent")
experiment = set_mlflow_experiment(projectConfig.mlflow_experiment_name)

# time_str = datetime.now(pytz.utc).astimezone(logging_timezone).strftime('%Y-%m-%d-%H:%M:%S-%Z')

with mlflow.start_run(experiment_id=experiment.experiment_id,
                                   run_name=f"rag_eval_now") as rag_evaluate_run:
    
    #here we will use the Mosaic AI Agent Evaluation framework to evaluate the RAG model
    result = mlflow.evaluate(
        model=f"runs:/{run.info.run_id}/chain",
        data=eval_data,
        model_type="databricks-agent"
    )


# COMMAND ----------

display(result.metrics)

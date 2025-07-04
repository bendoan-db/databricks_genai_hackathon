# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install -q -U -r requirements.txt
# MAGIC %pip install uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('./configs/agent.yaml', 'r') as file:
    config = yaml.safe_load(file)

#load global configs
databricks_config = config['databricks_config']
retriever_config = config['retriever_config']

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
mlflow_experiment=databricks_config['mlflow_experiment']
eval_table=databricks_config['eval_table_name']
model_name=databricks_config['model_name']
secret_scope=databricks_config['pat_token_secret_scope']
secret_key=databricks_config['pat_token_secret_key']

#load vs configs
vector_search_endpoint = retriever_config['vector_search_endpoint']
vector_search_index = retriever_config['vector_search_index']
embedding_model = retriever_config['embedding_model']

doc_agent_config = config["doc_agent_config"]
genie_agent_config = config["genie_agent_config"]
supervisor_config = config["supervisor_agent_config"]

# COMMAND ----------

import os
from dbruntime.databricks_repl_context import get_context

HOSTNAME = get_context().browserHostName
USERNAME = get_context().user

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope=secret_scope, key=secret_key)
os.environ['DATABRICKS_URL'] = get_context().apiUrl

# COMMAND ----------

os.environ['DATABRICKS_TOKEN']

# COMMAND ----------

import mlflow
from dbruntime.databricks_repl_context import get_context

experiment_fqdn = f"/Users/{get_context().user}/{mlflow_experiment}"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_fqdn)

if experiment:
    experiment_id = experiment.experiment_id
    # Create the experiment if it does not exist
else:
    experiment_id = mlflow.create_experiment(experiment_fqdn)

mlflow.set_experiment(experiment_fqdn)

# COMMAND ----------

# MAGIC %run ./02_genie_agent

# COMMAND ----------

from IPython.display import display, Image

display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

example_input = {
        "messages": [
            {
                "role": "user",
                "content": "What was the quick ratio for American Express in 2022, and what factors drove this?",
            }
        ]
    }

# COMMAND ----------

response = AGENT.predict(example_input)
print(response.messages[0].content)

# COMMAND ----------

# MAGIC %run ./03_deep_research_agent

# COMMAND ----------

from IPython.display import display, Image

display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

example_input = {
        "messages": [
            {
                "role": "user",
                "content": "What was the quick ratio for American Express in 2022, and what factors drove this?",
            }
        ]
    }

# COMMAND ----------

response = AGENT.predict(example_input)
print(response.messages[0].content)

# COMMAND ----------

for e in AGENT.predict_stream(example_input):
    print(e)
    print("\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate

# COMMAND ----------

eval_dataset = spark.table(f"{catalog}.{schema}.{eval_table}")

# COMMAND ----------

display(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Guidelines

# COMMAND ----------

structure = """The response must use clear, concise language and structures responses logically. It avoids jargon or explains technical terms when used."""

# COMMAND ----------

from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
    ChatAgentRequest
)

from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    Safety,
    Guidelines
)

from evaluation_utils.figure_correctness import figure_correctness

def my_predict_fn(messages): # the signature corresponds to the keys in the "inputs" dict
  return AGENT.predict(
    messages=[ChatAgentMessage(**message) for message in messages]
  )

# Run evaluation with predefined scorers
eval_results = mlflow.genai.evaluate(
    data=eval_dataset.toPandas(),
    predict_fn=my_predict_fn,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Safety(),
        figure_correctness,
        Guidelines(name="structure", guidelines=structure),
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
  DatabricksVectorSearchIndex,
  DatabricksServingEndpoint,
  DatabricksSQLWarehouse,
  DatabricksFunction,
  DatabricksGenieSpace,
  DatabricksTable,
  DatabricksUCConnection
)

with mlflow.start_run():
    logged_chain_info = mlflow.pyfunc.log_model(
        python_model=os.path.join(os.getcwd(), "03_deep_research_agent"),
        model_config=os.path.join(os.getcwd(), "configs/research_agent.yaml"), 
        name=model_name,  # Required by MLflow
        code_paths=[os.path.join(os.getcwd(), "vector_search_utils"), os.path.join(os.getcwd(), "supervisor_utils")],
        input_example=example_input,
        resources=[
        DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.{vector_search_index}"),
        DatabricksServingEndpoint(endpoint_name=doc_agent_config["llm_config"]["llm_endpoint_name"]),
        DatabricksServingEndpoint(endpoint_name=genie_agent_config["llm_config"]["llm_endpoint_name"]),
        DatabricksServingEndpoint(endpoint_name=supervisor_config["llm_config"]["llm_endpoint_name"]),
        DatabricksServingEndpoint(endpoint_name=embedding_model),
        DatabricksGenieSpace(genie_space_id=genie_agent_config["genie_space_id"]),
        DatabricksTable(table_name=f"{catalog}.{schema}.genie_income_statement"),
        DatabricksTable(table_name=f"{catalog}.{schema}.genie_balance_sheet")
        ],
        pip_requirements=["-r requirements.txt"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_chain_info.run_id}/{model_name}",
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents

agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=uc_registered_model_info.version,
    environment_vars={
        "DATABRICKS_URL": get_context().apiUrl,
        "DATABRICKS_TOKEN": dbutils.secrets.get(
            scope=secret_scope, key=secret_key
        ),
    },
)

# COMMAND ----------



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

import os
from dbruntime.databricks_repl_context import get_context

TOKEN = get_context().apiToken
HOSTNAME = get_context().browserHostName
USERNAME = get_context().user

os.environ['DATABRICKS_TOKEN'] = TOKEN
os.environ['DATABRICKS_URL'] = get_context().apiUrl

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

#load vs configs
vector_search_endpoint = retriever_config['vector_search_endpoint']
vector_search_index = retriever_config['vector_search_index']
embedding_model = retriever_config['embedding_model']

doc_agent_config = config["doc_agent_config"]
genie_agent_config = config["genie_agent_config"]
supervisor_config = config["supervisor_agent_config"]

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

# MAGIC %run ./01a_unstructured_retrieval_agent

# COMMAND ----------

from IPython.display import display, Image

display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

example_input = {
        "messages": [
            {
                "role": "user",
                "content": "What was AAPL's most popular product in 2022?",
            }
        ]
    }

# COMMAND ----------

response = AGENT.predict(example_input)
print(response.messages[0].content)

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

figure_correctness_guideline = """You are an impartial judge tasked with verifying the numerical accuracy of a generated response compared to a ground truth value. Your goal is to determine whether the generated number is numerically correct within two decimal places of the ground truth.

Evaluation Rules:
* Round both the generated value and the ground truth to two decimal places.
* If the two rounded values are exactly equal, the answer is Correct.
* If the rounded values differ, the answer is Incorrect."""

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
        Guidelines(name="figure_correctness_guideline", guidelines=figure_correctness_guideline),
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

with mlflow.start_run(run_name="sec_rag_doan"):
    logged_chain_info = mlflow.pyfunc.log_model(
        python_model=os.path.join(os.getcwd(), "01a_unstructured_retrieval_agent"),
        model_config=os.path.join(os.getcwd(), "configs/agent.yaml"), 
        artifact_path="agent",  # Required by MLflow
        code_paths=[os.path.join(os.getcwd(), "tools")],
        input_example=example_input,
        resources=[
        DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.{vector_search_index}"),
        DatabricksServingEndpoint(endpoint_name=doc_agent_config["llm_config"]["llm_endpoint_name"]),
        DatabricksServingEndpoint(endpoint_name=genie_agent_config["llm_config"]["llm_endpoint_name"]),
        DatabricksServingEndpoint(endpoint_name=supervisor_config["llm_config"]["llm_endpoint_name"]),
        DatabricksServingEndpoint(endpoint_name=embedding_model)
        ],
        pip_requirements=["-r requirements.txt"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate

# COMMAND ----------

eval_df = spark.sql("select * from doan.finance_bench.eval_data_gold")
display(eval_df)

# COMMAND ----------

eval_sample = eval_df.limit(10)

# COMMAND ----------

# MAGIC %run ./judge_examples

# COMMAND ----------

display(examples_df)

# COMMAND ----------

# set evaluation experiment
# mlflow.set_experiment("/Users/forrest.murray@databricks.com/sec_rag")

# global_guidelines = {
#     "figure_correctness": [
#         "If the question asks for a figure, the generated response should contain the correct figure that's listed in the ground truth. The figure should be accurate within 1 decimal point."
#     ],
# }


with mlflow.start_run(run_name="sec_rag_agent_doan_with_judge_examples"):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df.toPandas(),
        model=f"runs:/{logged_chain_info.run_id}/agent",  # replace `agent` with artifact_path that you used when calling log_model.  By default, this is `agent`.
        model_type="databricks-agent",
        evaluator_config={
            "databricks-agent": {
                "metrics": [
                    "chunk_relevance",
                    "context_sufficiency",
                    "correctness",
                    "safety",
                    "groundedness",
                    #"document_recall"
                ],
                "examples_df": examples_df
            }
        },
    )

# COMMAND ----------

eval_results.tables['eval_results']

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_chain_info.run_id}/agent",
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "doan"
schema = "finance_bench"
model_name = "sec_rag_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)

# COMMAND ----------



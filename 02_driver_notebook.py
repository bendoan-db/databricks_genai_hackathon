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

#load vs configs
vector_search_endpoint = retriever_config['vector_search_endpoint']
vector_search_index = retriever_config['vector_search_index']
embedding_model = retriever_config['embedding_model']

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

# MAGIC %run ./01b_genie_agent

# COMMAND ----------

from IPython.display import display, Image

display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

example_input = {
        "messages": [
            {
                "role": "user",
                "content": "How did APPL's operating margin change between 2020 and 2021? What factors contributed to this?",
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

from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    Safety,
)

# Run evaluation with predefined scorers
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=AGENT.predict,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Safety(),
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
        python_model=os.path.join(os.getcwd(), "01_sec_agent"),
        model_config=os.path.join(os.getcwd(), "config.yaml"),  # Chain configuration set in 00_config
        artifact_path="agent",  # Required by MLflow
        code_paths=[os.path.join(os.getcwd(), "tools")],
        input_example=example_input,
        resources=[
        DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.{vector_search_index}"),
        DatabricksServingEndpoint(endpoint_name="doan-gpt-4o"),
        DatabricksServingEndpoint(endpoint_name="doan-o3"),
        DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
        DatabricksServingEndpoint(endpoint_name="databricks-claude-3-7-sonnet"),
        DatabricksServingEndpoint(endpoint_name=embedding_model)
        ],
        pip_requirements=["-r requirements.txt"],
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
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



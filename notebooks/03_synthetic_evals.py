# Databricks notebook source
# MAGIC %md # Synthesize evaluations from documents
# MAGIC
# MAGIC This notebook shows how you can synthesize evaluations for an agent that uses document retrieval. It uses the `generate_evals_df` method that is part of the `databricks-agents` Python package.

# COMMAND ----------

# %pip install mlflow mlflow[databricks] databricks-agents
# dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import sys, os, yaml
sys.path.append(os.path.abspath('..'))
from configs.project import ProjectConfig

with open("../configs/project.yml", "r") as file:
    data = yaml.safe_load(file)

projectConfig = ProjectConfig(**data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load a delta table with prepared text chunks

# COMMAND ----------

# TODO: choose the correct index here most likely "id_1"
_config = projectConfig.vector_search_attributes["id_1"]
display(spark.table(_config.source_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC sample and load into a pandas dataframe.

# COMMAND ----------

import pyspark.sql.functions as F
input_df = spark.table(_config.source_table_name).select(F.col("content_chunked").alias("content"), F.col("doc_uri")).limit(10).toPandas()

# COMMAND ----------

display(input_df)

# COMMAND ----------

# MAGIC %md ## Documentation 
# MAGIC
# MAGIC The API is shown below. For more details, see the documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/synthesize-evaluation-set)).  
# MAGIC
# MAGIC API:
# MAGIC ```py
# MAGIC def generate_evals_df(
# MAGIC     docs: Union[pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
# MAGIC     *,
# MAGIC     num_evals: int,
# MAGIC     agent_description: Optional[str] = None,
# MAGIC     question_guidelines: Optional[str] = None,
# MAGIC ) -> pd.DataFrame:
# MAGIC     """
# MAGIC     Generate an evaluation dataset with questions and expected answers.
# MAGIC     Generated evaluation set can be used with Databricks Agent Evaluation
# MAGIC     AWS: (https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluate-agent.html)
# MAGIC     Azure: (https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/evaluate-agent).
# MAGIC
# MAGIC     :param docs: A pandas/Spark DataFrame with a text column `content` and a `doc_uri` column.
# MAGIC     :param num_evals: The number of questions (and corresponding answers) to generate in total.
# MAGIC     :param agent_description: Optional, a task description of the agent.
# MAGIC     :param question_guidelines: Optional guidelines to help guide the synthetic question generation. This is a free-form string that will
# MAGIC         be used to prompt the generation. The string can be formatted in markdown and may include sections like:
# MAGIC         - User Personas: Types of users the agent should support
# MAGIC         - Example Questions: Sample questions to guide generation
# MAGIC         - Additional Guidelines: Extra rules or requirements
# MAGIC     """
# MAGIC ```

# COMMAND ----------

# These documents can be a Pandas DataFrame or a Spark DataFrame with two columns: 'content' and 'doc_uri'.
# template code to specify input data
_ = pd.DataFrame.from_records(
    [
      {
        'content': f"""
            Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Java,
            Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set
            of higher-level tools including Spark SQL for SQL and structured data processing, pandas API on Spark for pandas
            workloads, MLlib for machine learning, GraphX for graph processing, and Structured Streaming for incremental
            computation and stream processing.
        """,
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/'
      },
      {
        'content': f"""
            Spark’s primary abstraction is a distributed collection of items called a Dataset. Datasets can be created from Hadoop InputFormats (such as HDFS files) or by transforming other Datasets. Due to Python’s dynamic nature, we don’t need the Dataset to be strongly-typed in Python. As a result, all Datasets in Python are Dataset[Row], and we call it DataFrame to be consistent with the data frame concept in Pandas and R.""",
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/quick-start.html'
      }
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC The following code block synthesizes evaluations from a DataFrame of documents.  
# MAGIC - The input can be a Pandas DataFrame or a Spark DataFrame.  
# MAGIC - The output DataFrame can be directly used with `mlflow.evaluate()`.

# COMMAND ----------

agent_description = """
The agent is a RAG chatbot that answers questions about financial data with access to a corpus of SEC filings.

The agent's objectives are to:
• Understand and accurately parse user queries related to company financial performance, SEC regulatory filings, and earnings data.
• Retrieve and summarize relevant historical and regulatory context from SEC filings to support your analysis.
• Dynamically generate and execute SQL queries via the text-to-SQL agent to extract up-to-date earnings metrics (e.g., EPS, revenue, net income) from the data warehouse.
• Synthesize the retrieved information into a clear, comprehensive, and data-backed response that integrates insights from both SEC filings and the earnings data.
• Ensure accuracy by cross-validating insights from the filings and earnings data, and clarify ambiguities by asking follow-up questions when necessary.
• Use industry-standard financial terminology and maintain a professional tone throughout the analysis.

Workflow:
1. Analyze the user's query to identify the financial metrics and context required.
2. Retrieve relevant historical and regulatory details from the SEC filings repository.
3. Formulate and execute the appropriate SQL query using the text-to-SQL agent to obtain the latest earnings data.
4. Integrate findings from sources into a cohesive, insightful answer with proper data citations.
"""

# COMMAND ----------

question_guidelines = """
# User personas
- A financial analyst with a background in finance and accounting.
- A financial journalist with a background in finance and economics.
- A financial researcher with a background in finance and economics.

# Example questions
- What was Walmart's best quarter in 2022?
- What was 3M's net income (loss) attributable to common shareholders for the six months ended June 30, 2023?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

# COMMAND ----------

projectConfig.llm_endpoint_names[0]

# COMMAND ----------

import mlflow
from databricks.agents.evals import generate_evals_df
import pandas as pd


num_evals = 10

evals = generate_evals_df(
    input_df,
    # The total number of evals to generate. The method attempts to generate evals that have full coverage over the documents
    # provided. If this number is less than the number of documents, some documents will not have any evaluations generated. 
    # For details about how `num_evals` is used to distribute evaluations across the documents, 
    # see the documentation: 
    # AWS: https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html#num-evals. 
    # Azure: https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/synthesize-evaluation-set 
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    agent_description=agent_description,
    question_guidelines=question_guidelines
)

display(evals)

# COMMAND ----------

from src.utils import set_mlflow_experiment

experiment = set_mlflow_experiment(projectConfig.mlflow_experiment_name)

# COMMAND ----------

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
with mlflow.start_run(run_name="Synthetic Evaluation"):
  results = mlflow.evaluate(
    model=f"endpoints:/{projectConfig.llm_endpoint_names[0]}",
    data=evals,
    model_type="databricks-agent"
  )

display(results.tables['eval_results'])

# Note: To use a different model serving endpoint, use the following snippet to define an agent_fn. Then, specify that function using the `model` argument.
# MODEL_SERVING_ENDPOINT_NAME = '...'
# def agent_fn(input):
#   client = mlflow.deployments.get_deploy_client("databricks")
#   return client.predict(endpoint=MODEL_SERVING_ENDPOINT_NAME, inputs=input)

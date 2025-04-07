# Agent Monitoring Overview
Databricks offers several ways to monitoring agents and LLMs deployed in Databricks. The notebooks in this repository highlight various monitoring patterns we offer for different LLM-based applications.

## [Lakehouse Monitoring for GenAI](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring)
Lakehouse Monitoring for GenAI uses MLflow Tracing, an open standard for GenAI observability based on Open Telemetry, to instrument and capture production logs from your GenAI app. To use monitoring, first instrument your GenAI app with MLflow Tracing. 
 - For any agentic applications deployed with the [Databricks Agent Framework](https://docs.databricks.com/aws/en/generative-ai/tutorials/agent-framework-notebook) and `ChatAgent()` monitoring is deployed automatically with `agents.deploy()` and is viewable in the experiment where you logged the agent code.

## [Monitoring for Externally Deployed Agents ](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring-non-agent-framework)
- To monitor agents deployed outside of Databricks, you will need to instrument your code with [MLFlow traces](https://mlflow.org/docs/latest/tracing?utm_source=google&utm_medium=cpc&utm_campaign=PMax-Website-Traffic-Tracing&utm_content=Tracing&utm_term=PMax&utm_source=google&utm_medium=cpc&utm_term=&utm_campaign=&gad_source=1&gclid=Cj0KCQjw782_BhDjARIsABTv_JCB-HDhccqP_NcPghuM-LNq-bkt_u1-Rvh9qWRieCo5R-99RDkNzVMaArNVEALw_wcB) and set the traces to be logged to Databricks.
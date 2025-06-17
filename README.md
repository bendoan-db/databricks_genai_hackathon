# databricks_genai_hackathon

# Overview

This repository offers complete examples of implementing GenAI agents on Databricks following the [LangGraph supervisor multiagent architecture](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/). It leverages the Databricks Agent Framework, integrates with [Langchain/Langgraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/), and supports Vector Search, Unity Catalog Functions, and [Genie](https://www.databricks.com/product/ai-bi/genie) — a state-of-the-art text-to-SQL tool developed by Databricks.

The agents are deployed using Databricks Model Serving and are monitored through Databricks Model Monitoring. The repository includes small datasets featuring preprocessed text chunks from publicly available SEC filings, mock structured data for select companies, and an evaluation set of question/answer pairs sourced from the FinanceBench repository. This evaluation data is used for both offline and online performance assessment.


# Notebooks

The project is organized into multiple components including notebooks, configs, agents, and environment setup.

Users have to edit the YAML files for the project and for the individual Agents that get deployed on Databricks. The models and the contained agents are specified in notebooks that users can run as is or modify to achieve the specific goals of the hackathon.

- `00_driver_notebook` serves as the primary runner to test and evaluate agent code. It contains code to test the Agent on an example input, run a GenAI evaluation, and deploy using the agent framework
- `01x_unstructured_retrieval_agent` has code for a supervisor and document agent graph, whose primary task is to use a retrieval tools. The graph is implemented with LangGraph and ChatAgent frameworks
- `02_genie_agent` has code for a supervisor, document agent, and sql agent graph
- `03_deep_research_agent` has all of the code in 02_genie_agent, as well as a planning agent. It also has its own config file under `configs`


# Cluster Config

On Databricks, use either a serverless cluster or a standard cluster running Runtime 15.4 LTS or higher. The Machine Learning Runtime is not recommended.

If you’re using a standard Databricks Runtime, please [install](https://docs.databricks.com/aws/en/libraries/cluster-libraries) the required libraries listed in the [requirements.txt](requirements.txt) file. In this case, you can omit the `pip install ...` commands at the beginning of the notebooks.

If you’re using Serverless compute, please uncomment and run the `pip install ...` commands in each notebook to install the necessary libraries.


# For admins

- ideally, hackathon users should be granted permission to create their individual unity catalog schema. This greatly reduces the need to specify individual assets like tablenames, uc-function names, models etc. 
- caution when cloning the repo to individual users workspace folders: yaml files do not get cloned, users have to copy and edit them manually 

# Project Setup

 1. edit the configs in the `configs` directory to specify the parameters for the various agents in this repo
  - `ingestion_config` contains configurations to run the notebooks in the `document_ingestion` directory. This dir contains code to ingest the sample data in `data` folder. You can use this in place of your own data for testing if desired
  - `agent.yaml` contains agent configurations for the agents in the `01x_unstructured_retrieval_agent` and `02_genie_agent` notebooks
  - `research_agent.yaml` contains agent configurations for the agents in the `03_deep_research_agent` notebooks
 2. Modify the Agent code in any of the `01`-`03` notebooks as necessary
 3. Test your modifications in the `00_driver` notebook
  
 
# Disclaimer

These examples are provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors, copyright holders, or contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The authors and maintainers of this repository make no guarantees about the suitability, reliability, availability, timeliness, security or accuracy of the software. It is your responsibility to determine that the software meets your needs and complies with your system requirements.

No support is provided with this software. Users are solely responsible for installation, use, and troubleshooting. While issues and pull requests may be submitted, there is no guarantee of response or resolution.

By using this software, you acknowledge that you have read this disclaimer, understand it, and agree to be bound by its terms.

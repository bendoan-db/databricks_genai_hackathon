import functools
import os
from typing import Any, Generator, Literal, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel

mlflow.langchain.autolog()

###################################################
## Create a GenieAgent with access to a Genie Space
###################################################


multi_agent_config = mlflow.models.ModelConfig(development_config="../configs/langgraph_multiagent_genie_pat.yaml")

# TODO add GENIE_SPACE_ID and a description for this space
GENIE_SPACE_ID = multi_agent_config.get("genie_space_id")
# GENIE_SPACE_ID = "01efe9685e931086a75d4bb913f22b8e"
genie_agent_description = """
The text2sql agent is a specialized tool designed to interpret natural language queries and translate them into SQL commands to access financial data from companies with SEC filings. It works with two key tables:
	•	Balance Sheet Data: This table contains detailed records of a company’s assets, liabilities, and shareholders’ equity as reported in their SEC filings. It allows users to retrieve information on financial position, analyze liquidity ratios, and understand the structure of a company’s financing.
	•	Income Statement Data: This table holds information on revenue, expenses, and net income, providing insights into a company’s operational performance. Users can query trends in sales, profit margins, and cost structures, which are essential for evaluating the company’s profitability and operational efficiency.

By leveraging the agent, users can seamlessly ask questions such as, “Show me the total liabilities for companies with revenue over $500M” or “List companies with a debt-to-equity ratio above 1.5 based on their most recent SEC filings.” The agent then dynamically generates the corresponding SQL query to extract and deliver the requested information, making financial analysis both intuitive and efficient.
"""

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=genie_agent_description,
    # DB_MODEL_SERVING_HOST_URL is set on an agent endpoints but doesn't exist in the notebook
    client=WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST") or os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)


############################################
# Define your LLM endpoint and system prompt
############################################

# TODO: Replace with your model serving endpoint, multi-agent Genie works best with GPT 4o and GPT o1 models.
LLM_ENDPOINT_NAME = multi_agent_config.get("multi_agent_llm_config").get("llm_endpoint_name")
assert LLM_ENDPOINT_NAME is not None
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


############################################################
# Create a code agent
# You can also create agents with access to additional tools
############################################################
client = DatabricksFunctionClient()
set_uc_function_client(client)

tools = []

# TODO if desired, add additional tools and update the description of this agent
uc_tool_names = ["system.ai.*"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)
code_agent_description = (
    "The Coder agent specializes in solving programming challenges, generating code snippets, debugging issues, and explaining complex coding concepts.",
)
code_agent = create_react_agent(llm, tools=tools)

#############################
# Define the supervisor agent
#############################

worker_descriptions = {
    "Genie": genie_agent_description,
    "Coder": code_agent_description,
}

formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
options = ["FINISH"] + list(worker_descriptions.keys())


def supervisor_agent(state):
    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    return supervisor_chain.invoke(state)


#######################################
# Define our multiagent graph structure
#######################################


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": name,
            }
        ]
    }


def final_answer(state):
    system_prompt = "Using only the content in the messages, respond to the user's question using the answer given by the other agents."
    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    final_answer_chain = preprocessor | llm
    return {"messages": [final_answer_chain.invoke(state)]}


class AgentState(ChatAgentState):
    next_node: str


code_node = functools.partial(agent_node, agent=code_agent, name="Coder")
genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")

workflow = StateGraph(AgentState)
workflow.add_node("Genie", genie_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("supervisor")
# We want our workers to ALWAYS "report back" to the supervisor when done
for worker in worker_descriptions.keys():
    workflow.add_edge(worker, "supervisor")

# Let the supervisor decide which next node to go
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
)
workflow.add_edge("final_answer", END)
multi_agent = workflow.compile()

###################################
# Wrap our multi-agent in ChatAgent
###################################


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg})
                    for msg in node_data.get("messages", [])
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)

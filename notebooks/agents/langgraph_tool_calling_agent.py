from typing import Any, Generator, Optional, Sequence, Union

import mlflow
import uuid
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
multi_agent_config = mlflow.models.ModelConfig(development_config="../configs/project.yml")
LLM_ENDPOINT_NAME = multi_agent_config.get("llm_endpoint_names")[0]
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = """
You are a RAG (Retrieval-Augmented Generation) agent designed for financial data analysis with dual data access:
1. A comprehensive repository of SEC filings.
2. A text-to-SQL agent that queries company earnings data stored in our data warehouse tables.

Your objectives are to:
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
4. Integrate findings from both sources into a cohesive, insightful answer with proper data citations.
5. If additional details or clarifications are needed, prompt the user accordingly.

Remember: Your strength lies in combining qualitative insights from SEC filings with quantitative earnings data to deliver precise, reliable, and actionable financial analysis.
"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
# Below, we add the `system.ai.python_exec` UDF, which provides
# a python code interpreter tool to our agent
# You can also add local LangChain python tools. See https://python.langchain.com/docs/concepts/tools

# TODO: Add additional tools
uc_tool_names = ["system.ai.python_exec", f"{multi_agent_config.get('uc_catalog')}.{multi_agent_config.get('uc_schema')}.lookup_ticker_info"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# for details

# TODO: Add vector search indexes
def get_index_name(config: mlflow.models.model_config.ModelConfig, id: str):
    _index_name = f"{config.get('uc_catalog')}.{config.get('uc_schema')}.{config.get('vector_search_attributes').get(id).get('table_name')}_index"
    index_name = (
        config.get("vector_search_attributes").get(id).get("index_name", _index_name)
    )
    return index_name


index_name = get_index_name(multi_agent_config, "id_1")

vector_search_tools = [
        VectorSearchRetrieverTool(
        index_name=index_name,
        # filters="..."
    )
]
tools.extend(vector_search_tools)

#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    # def predict_stream(
    #     self,
    #     messages: list[ChatAgentMessage],
    #     context: Optional[ChatContext] = None,
    #     custom_inputs: Optional[dict[str, Any]] = None,
    # ) -> Generator[ChatAgentChunk, None, None]:
    #     request = {"messages": self._convert_messages_to_dict(messages)}
    #     for event in self.agent.stream(request, stream_mode="updates"):
    #         for node_data in event.values():
    #             yield from (
    #                 ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
    #             )

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        response_id = str(uuid.uuid4())
        
        for event in self.agent.stream(request, stream_mode="messages"):
            # Event is a tuple: (AIMessageChunk, metadata)
            if isinstance(event, tuple) and len(event) >= 2:
                message_chunk, metadata = event[0], event[1]
                # Extract content from AIMessageChunk
                content = message_chunk.content
                idid = message_chunk.id
                # AIMessageChunk typically doesn’t have role in stream_mode="messages", default to "assistant"
                role = getattr(message_chunk, "role", "assistant") if hasattr(message_chunk, "role") else "assistant"
            else:
                print("Unexpected event format:", event)
                continue
            
            if not content:  # Skip empty chunks
                continue

            response_id = str(uuid.uuid4())

            chunk = ChatAgentChunk(
                delta=ChatAgentMessage(
                        **{
                            "role": role,
                            "content": content,
                            "id": response_id,
                        }
                    )
            )
            yield chunk



# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)

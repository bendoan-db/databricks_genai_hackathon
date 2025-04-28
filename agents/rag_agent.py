from typing import Any, Generator, Optional, Sequence, Union

import mlflow
import uuid
from databricks_langchain import (
    ChatDatabricks,
    # UCFunctionToolkit,
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

from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
multi_agent_config = mlflow.models.ModelConfig(development_config="../configs/rag_agent.yaml")
LLM_ENDPOINT_NAME = multi_agent_config.get("rag_agent_llm_config").get("llm_endpoint_name")
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, 
                    **multi_agent_config.get("rag_agent_llm_config").get("llm_parameters")
                    )

# TODO: Update with your system prompt
system_prompt = multi_agent_config.get("rag_agent_llm_config").get("system_prompt")

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
uc_tool_names = multi_agent_config.get("uc_tool_names")
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names, client=client)
tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# for details

# TODO: Add vector search indexes
index_name = multi_agent_config.get("retriever_config").get("vector_search_index")

vector_search_tools = [
        VectorSearchRetrieverTool(
        index_name=index_name,
        tool_description=multi_agent_config.get("retriever_config").get("tool_description"),
        num_results=multi_agent_config.get("retriever_config").get("parameters").get("num_results"),
        query_type=multi_agent_config.get("retriever_config").get("parameters").get("query_type"), # "HYBRID"
        # filters="...",
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

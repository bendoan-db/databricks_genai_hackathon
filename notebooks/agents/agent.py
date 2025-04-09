from typing import Any, Generator, Optional, Sequence, Union
import functools

import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain.sql_database import SQLDatabase
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent, create_react_agent

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME =  "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


system_prompt = """
You are a helpful assistant. You can assist with help answiring questions about PySpark.
Use tool 'SQLTools" if user ask you about SQL query or analytics question.
Use tools 'VectorSearch' if user ask you about a specific document or a specific page.
"""

tools = []


claude="databricks-claude-3-7-sonnet"
llama31="databricks-meta-llama-3-1-70b-instruct"
db = SQLDatabase.from_databricks(catalog="gportier_demo", schema="spark_dock", engine_args={"pool_pre_ping": True,})
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)



vector_search_tools = [
         VectorSearchRetrieverTool(
         index_name="gportier_demo.rag_chatbot.databricks_documentation_vs_index",
         name="VectorSearchRetrieverTool"  # Forcer le nom

         # filters="..."
     )
 ]

sql_tools = toolkit.get_tools()
tools.extend(sql_tools)
tools.extend(vector_search_tools)


#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools,
    agent_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.get("tool_calls"):
            # TODO: Is there a better way than listing all tools func?
            tool_call = last_message["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            if tool_name in ["sql_db_list_tables", "sql_db_schema", "sql_db_query", "sql_db_query_checker"]:
                return "sql"
            elif tool_name == "gportier_demo__rag_chatbot__databricks_documentation_vs_index":
                return "vector"
            else:
                print(f"Unknown tool: {tool_name}")
                return "end"
        return "end"

    if agent_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": agent_prompt}]
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

    def add_custom_outputs(state: ChatAgentState):
        # TODO: Return extra content with the custom_outputs key before returning
        return {
            "custom_outputs": {
                **(state.get("custom_outputs") or {}),
                **(state.get("custom_inputs") or {}),
                "key": "value",
            }, 
        }

    
        
    def format_response(state: ChatAgentState):
        # TODO DO BETTER
        last_message = next((msg for msg in reversed(state["messages"]) if msg["role"] == "assistant"), None)
        if last_message:
            content = last_message["content"]
            formatted_content = f"**{content}     !!!**"
            return {
                "messages": [{"role": "assistant", "content": formatted_content}]
            }
        
        return {"messages": []}  
      
    def format_response(state: ChatAgentState):
        user_question = next((msg["content"] for msg in reversed(state["messages"]) if msg["role"] == "user"), "No question found")
        assistant_answer = next((msg["content"] for msg in reversed(state["messages"]) if msg["role"] == "assistant"), "No answer found")
        
        markdown_template = f"""**{user_question}**

                            # Answer
                            **{assistant_answer}!!!**
                            """

        return {
            "messages": [{"role": "assistant", "content": markdown_template}]
        }




    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("SQLTools", ChatAgentToolNode(sql_tools))
    workflow.add_node("VectoreSearch", ChatAgentToolNode(vector_search_tools))
    workflow.add_node("add_custom_outputs", RunnableLambda(add_custom_outputs))
    workflow.add_node("format_response", RunnableLambda(format_response))  
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "sql": "SQLTools",
            "vector":"VectoreSearch",
            "end": "add_custom_outputs",
        },
    )

    workflow.add_edge("SQLTools", "agent")
    workflow.add_edge("VectoreSearch", "agent")

    workflow.add_edge("add_custom_outputs", "format_response")
    workflow.add_edge("format_response", END)

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
        # TODO: Use context and custom_inputs to alter the behavior of the agent
        request = {
            "messages": self._convert_messages_to_dict(messages),
            **({"custom_inputs": custom_inputs} if custom_inputs else {}),
            **({"context": context.model_dump_compat()} if context else {}),
        }

        response = ChatAgentResponse(messages=[])
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                if not node_data:
                    continue
                for msg in node_data.get("messages", []):
                    response.messages.append(ChatAgentMessage(**msg))
                if "custom_outputs" in node_data:
                    response.custom_outputs = node_data["custom_outputs"]
        return response

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        # TODO: Use context and custom_inputs to alter the behavior of the agent
        request = {
            "messages": self._convert_messages_to_dict(messages),
            **({"custom_inputs": custom_inputs} if custom_inputs else {}),
            **({"context": context.model_dump_compat()} if context else {}),
        }

        last_message = None
        last_custom_outputs = None

        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                if not node_data:
                    continue
                messages = node_data.get("messages", [])
                custom_outputs = node_data.get("custom_outputs")

                for message in messages:
                    if last_message:
                        yield ChatAgentChunk(delta=last_message)
                    last_message = message
                if custom_outputs:
                    last_custom_outputs = custom_outputs
        if last_message:
            yield ChatAgentChunk(delta=last_message, custom_outputs=last_custom_outputs)


agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)

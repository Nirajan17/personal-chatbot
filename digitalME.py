import os
from dotenv import load_dotenv
import getpass
import time
import warnings
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import START, END, StateGraph
import operator
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

warnings.filterwarnings(action="ignore")

# Load environment variables
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY") or getpass.getpass("Enter your Pinecone API key: ")

# Initialize Groq LLM
from langchain_groq import ChatGroq

groqLLM = ChatGroq(
    model="deepseek-r1-distill-qwen-32b",  # Assuming this is the correct Grok model name from xAI; adjust as needed
    api_key=api_key,
    temperature=0.7,
    max_tokens=512  # Limit response length
)

# Initialize embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
index_name = "personal"

# Create or connect to Pinecone index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize Pinecone vector store
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load documents into vector store
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

def custom_loader(file_path: str):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# loader = DirectoryLoader("personal", glob="**/*", show_progress=True, loader_cls=custom_loader)
# docs = loader.load()

# Ensure vector store is populated
# if not index.describe_index_stats()['total_vector_count']:
#     print("Populating vector store with documents...")
#     vector_store.add_documents(docs)

# Set up retriever
groq_retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# Define tools
from langchain_core.tools import tool

@tool
def file_reader(file_path: str) -> str:
    """Read content from a text file."""
    try:
        cwd = os.getcwd()  # Get current working directory
        print(f"Looking for file in: {cwd}")  # Debug print
        if not os.path.exists(file_path):
            return f"Error: File not found at path {file_path} (checked in {cwd})"
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Tool output: {content}")  # Debug print
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
@tool
def write_to_file(file_path: str, content: str, append: bool = False) -> str:
    """
    Write or append content to a specified file.
    
    Args:
        file_path (str): The path to the file.
        content (str): The content to write into the file.
        append (bool): If True, appends to the file; if False, overwrites it.
    
    Returns:
        str: Confirmation message or error if it fails.
    """
    try:
        mode = "a" if append else "w"
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"Successfully {'appended to' if append else 'wrote to'} {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"
    
class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///transcript.db")

from langchain_core.tools import tool
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# assert len(query_prompt_template.messages) == 1

@tool
def sql_database_tool(user_query: str) -> dict:
    """Generate and execute an SQL query based on the user's query."""
    # Step 1: Generate the SQL query
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": user_query,
        }
    )
    structured_groq_llm = groqLLM.with_structured_output(QueryOutput)
    query_result = structured_groq_llm.invoke(prompt)
    sql_query = query_result["query"]

    # Step 2: Execute the SQL query
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    sql_result = execute_query_tool.invoke(sql_query)

    return sql_result

tools = [file_reader, write_to_file, sql_database_tool]
groqLLM_with_tools = groqLLM.bind_tools(tools)

# Define prompt template
from langchain.prompts import ChatPromptTemplate

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

template = """
You are digitalME, an AI assistant created by Berojgar & company, designed to engage in natural, helpful conversations. You can answer general questions, provide insights, and assist with tasks. When relevant, you have access to a vector database of documents and tools to enhance your responses.

### Chat History:
{messages}

### User Query:
{user_query}

### Retrieved Documents (from vector database, if applicable):
{retrieved_documents}

### SQL Result (from database query, if applicable):
{sql_result}

### Instructions:
- Engage in a natural, conversational tone as a helpful assistant.
- Use the chat history to maintain context and provide coherent, relevant responses.
- If the user query can be answered directly based on general knowledge or chat history, do so concisely and clearly.
- If the query relates to information in the retrieved documents, incorporate that information into your response and briefly note that it came from the documents (e.g., "Based on your documents...").
- If the query requires additional information or functionality (e.g., reading a file, writing to a file, or querying a database):
  - For file-related tasks, use the available tools, execute them fully, and include their results in your response (e.g., "I used the file_reader tool to check your sports from sports.txt, and here's what I found: [result]", or "I have successfully written [text] in the file [file_name]").
  - For database-related queries, use the `sql_database_tool` by passing the exact user query as `user_query` (e.g., call `sql_database_tool` with `user_query` if the question involves data like students, subjects, or other database-stored info). Summarize the tool’s output in your response (e.g., "I checked the database, and the total subjects are [sql_result]"), without outputting raw JSON tool calls.
- If a tool returns an error (e.g., file not found or database issue), report it clearly in the response (e.g., "I tried using the file_reader tool, but the file sports.txt wasn’t found", "I couldn’t write to [file_name]", or "There was an issue querying the database, so I can’t fetch that data right now").
- If `sql_result` is available from a previous database query, incorporate it into your response naturally (e.g., "Based on the database, [sql_result]").
- If no retrieved documents, tools, or SQL results are relevant, proceed with a general conversational response.
- Keep your answers friendly, concise, and tailored to the user's intent.

### Response:
Provide your conversational response here, blending general knowledge, document insights, tool outputs, and database results as needed.
"""

prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_template(template)

# Set up LangGraph workflow
class GraphState(TypedDict):
    user_query : str
    retrieved_documents : Sequence[str]
    messages: Annotated[list, add_messages]
    sql_result: str


def input_node(state: GraphState) -> GraphState:
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=state["user_query"])]
    return state

def retrieval_node(state: GraphState)->GraphState:
    retrieved_documents = groq_retriever.invoke(state['user_query'])
    state['retrieved_documents'] = [doc.page_content for doc in retrieved_documents]
    return state

def processing_node(state: GraphState)->GraphState:
    chain = prompt | groqLLM_with_tools
    response = chain.invoke(
        {
            "messages": state["messages"],
            "user_query": state['user_query'],
            "retrieved_documents": state['retrieved_documents'],
            "sql_result": state.get("sql_result", "")
        }
    )
    # state["messages"] = state["messages"] + [response]
    if response.tool_calls:
        state["messages"] = state["messages"] + [response]
    else:
        state["messages"] = state["messages"] + [response]
        if state.get("sql_result"):
            state["messages"].append(AIMessage(content=f"Result: {state['sql_result']}"))
    return state

tool_node = ToolNode(tools=tools)

workflow = StateGraph(GraphState)

workflow.add_node("input", input_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("processing", processing_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "input")
workflow.add_edge("input","retrieval")
workflow.add_edge("retrieval","processing")

workflow.add_conditional_edges("processing", tools_condition)  # if tool call is there, it is called
workflow.add_edge("tools", "processing")  # if toolcall is made then tool execution should occur, if occur, this is ..

workflow.add_edge("processing",END)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

# Chat loop with error handling
config = {"configurable": {"thread_id": "1"}}

# print("Start chatting! Type 'exit' to quit.")
config = {"configurable": {"thread_id": "1"}} 

print("Start chatting! Type 'exit' to quit.")
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Goodbye!")
        break
    
    initial_state = {
        "user_query": user_query,
        "retrieved_documents": [],
        "messages": app.get_state(config).values.get("messages", []),  # Load previous messages
        "sql_result": ""
    }
    
    result = app.invoke(initial_state, config=config)
    print("Assistant:", result["messages"][-1].content)
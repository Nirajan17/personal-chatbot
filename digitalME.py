import os
from dotenv import load_dotenv
import getpass
import time
import warnings
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import START, END, StateGraph
import operator
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import warnings
from transformers import logging as transformers_logging

# Disable Hugging Face warnings
# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
transformers_logging.set_verbosity_error()

warnings.filterwarnings(action="ignore")

# Load environment variables
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY") or getpass.getpass("Enter your Pinecone API key: ")

# Initialize Groq LLM
from langchain_groq import ChatGroq

groq_llm = ChatGroq(
    model="deepseek-r1-distill-qwen-32b",
    api_key=api_key,
    temperature=0.7,
    max_tokens=512
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
    print(f"Created new Pinecone index: {index_name}")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
        print("Waiting for index to be ready...")

index = pc.Index(index_name)

# Initialize Pinecone vector store
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load documents into vector store - uncomment when needed
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

def custom_loader(file_path: str):
    """Select appropriate loader based on file extension."""
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Uncomment to load documents
# loader = DirectoryLoader("personal", glob="**/*", show_progress=True, loader_func=custom_loader)
# docs = loader.load()
# 
# # Ensure vector store is populated
# if index.describe_index_stats()['total_vector_count'] == 0:
#     print("Populating vector store with documents...")
#     vector_store.add_documents(docs)
#     print(f"Added {len(docs)} documents to vector store")

# Set up retriever
groq_retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# Define tools
from langchain_core.tools import tool

@tool
def file_reader(file_path: str) -> str:
    """Read content from a text file."""
    try:
        cwd = os.getcwd()
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
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
    query: Annotated[str, "Syntactically valid SQL query."]

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///transcript.db")

from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

@tool
def sql_database_tool(user_query: str) -> str:
    """Generate and execute an SQL query based on the user's query."""
    try:
        # Step 1: Generate the SQL query
        sql_prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": user_query,
            }
        )
        structured_groq_llm = groq_llm.with_structured_output(QueryOutput)
        query_result = structured_groq_llm.invoke(sql_prompt)
        sql_query = query_result["query"]

        # Step 2: Execute the SQL query
        from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        sql_result = execute_query_tool.invoke(sql_query)

        return sql_result
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

tools = [file_reader, write_to_file, sql_database_tool]
groq_llm_with_tools = groq_llm.bind_tools(tools)

# Define prompt template
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
  - For database-related queries, use the `sql_database_tool` by passing the exact user query as `user_query` (e.g., call `sql_database_tool` with `user_query` if the question involves data like students, subjects, or other database-stored info). Summarize the tool's output in your response (e.g., "I checked the database, and the total subjects are [sql_result]"), without outputting raw JSON tool calls.
- If a tool returns an error (e.g., file not found or database issue), report it clearly in the response (e.g., "I tried using the file_reader tool, but the file sports.txt wasn't found", "I couldn't write to [file_name]", or "There was an issue querying the database, so I can't fetch that data right now").
- If `sql_result` is available from a previous database query, incorporate it into your response naturally (e.g., "Based on the database, [sql_result]").
- If no retrieved documents, tools, or SQL results are relevant, proceed with a general conversational response.
- Keep your answers friendly, concise, and tailored to the user's intent.

### Response:
Provide your conversational response here, blending general knowledge, document insights, tool outputs, and database results as needed.
"""

prompt = ChatPromptTemplate.from_template(template)

# Set up LangGraph workflow
class GraphState(TypedDict):
    user_query: str
    retrieved_documents: Sequence[str]
    messages: Annotated[list[BaseMessage], add_messages]
    sql_result: Optional[str]

def input_node(state: GraphState) -> GraphState:
    """Initialize messages if not present."""
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=state["user_query"])]
    else:
        state["messages"].append(HumanMessage(content=state["user_query"]))
    return state

def retrieval_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents from vector store."""
    try:
        retrieved_documents = groq_retriever.invoke(state['user_query'])
        state['retrieved_documents'] = [doc.page_content for doc in retrieved_documents]
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        state['retrieved_documents'] = []
    return state

# def processing_node(state: GraphState) -> GraphState:
#     """Process user query and generate response."""
#     chain = prompt | groq_llm_with_tools
#     response = chain.invoke(
#         {
#             "messages": state["messages"],
#             "user_query": state['user_query'],
#             "retrieved_documents": state['retrieved_documents'],
#             "sql_result": state.get("sql_result", "")
#         }
#     )
    
#     # Add response to messages
#     state["messages"].append(response)
#     return state

def processing_node(state: GraphState)->GraphState:
    chain = prompt | groq_llm_with_tools
    response = chain.invoke(
        {
            "messages": state["messages"],
            "user_query": state['user_query'],
            "retrieved_documents": state['retrieved_documents'],
            "sql_result": state.get("sql_result", "")
        }
    )
    
    # Check if there are tool calls
    if response.tool_calls:
        # Add the assistant's message with the tool call intent
        state["messages"].append(response)
        # Note: The tool results will be handled by the tool_node and will come back
        # to this node for final processing
    else:
        # This is either a regular response or the final response after tool execution
        # We need to filter out any tool messages that were previously in the state
        filtered_messages = []
        for msg in state["messages"]:
            # Only keep messages that aren't raw tool outputs
            if not (hasattr(msg, 'type') and msg.type == 'tool'):
                filtered_messages.append(msg)
        
        # Add the final response
        filtered_messages.append(response)
        state["messages"] = filtered_messages
        
    return state

# Configure the workflow
tool_node = ToolNode(tools=tools)
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("input", input_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("processing", processing_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "input")
workflow.add_edge("input", "retrieval")
workflow.add_edge("retrieval", "processing")
workflow.add_conditional_edges("processing", tools_condition)
workflow.add_edge("tools", "processing")
workflow.add_edge("processing", END)

# Set up memory for conversation history
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Chat loop
def start_chat():
    """Run the chat interface."""
    config = {"configurable": {"thread_id": "1"}}
    print("Start chatting! Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        try:
            # Get current state or create new state
            current_state = app.get_state(config)
            
            initial_state = {
                "user_query": user_query,
                "retrieved_documents": [],
                "messages": current_state.values.get("messages", []),
                "sql_result": ""
            }
            
            result = app.invoke(initial_state, config=config)
            print("Assistant:", result["messages"][-1].content)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Assistant: I encountered an error processing your request. Please try again.")

if __name__ == "__main__":
    start_chat()
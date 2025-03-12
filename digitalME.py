import os
from dotenv import load_dotenv
import getpass
import time
import warnings

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

tools = [file_reader]
groqLLM_with_tools = groqLLM.bind_tools(tools)

# Define prompt template
from langchain.prompts import ChatPromptTemplate

template = """
You are digitalME, an AI assistant created by Berojgar & company, designed to engage in natural, helpful conversations. You can answer general questions, provide insights, and assist with tasks. When relevant, you have access to a vector database of documents and tools to enhance your responses.

### Chat History:
{messages}

### User Query:
{user_query}

### Retrieved Documents (from vector database, if applicable):
{retrieved_documents}

### Instructions:
- Engage in a natural, conversational tone as a helpful assistant.
- Use the chat history to maintain context and provide coherent, relevant responses.
- If the user query can be answered directly based on general knowledge or chat history, do so concisely and clearly.
- If the query relates to information in the retrieved documents, incorporate that information into your response and briefly note that it came from the documents (e.g., "Based on your documents...").
- If the query requires additional information or functionality (e.g., reading a file), use the available tools, execute them fully, and include their results in your response. Do NOT output raw JSON tool calls; instead, summarize the tool's output (e.g., "I used the file_reader tool to check your sports from sports.txt, and here's what I found: [result]").
- If a tool returns an error (e.g., file not found), report it clearly in the response (e.g., "I tried using the file_reader tool, but the file sports.txt wasn’t found").
- If no retrieved documents or tools are relevant, proceed with a general conversational response.
- Keep your answers friendly, concise, and tailored to the user's intent.

### Response:
Provide your conversational response here, blending general knowledge, document insights, and tool outputs as needed.
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up LangGraph workflow
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class GraphState(TypedDict):
    user_query: str
    retrieved_documents: Sequence[str]
    messages: Annotated[list, add_messages]

def input_node(state: GraphState) -> GraphState:
    """Initialize state with user query as a message if no messages exist."""
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=state["user_query"])]
    return state

def retrieval_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents from Pinecone."""
    retrieved_documents = groq_retriever.invoke(state['user_query'])
    state['retrieved_documents'] = [doc.page_content for doc in retrieved_documents]  # Extract text content
    return state

def processing_node(state: GraphState) -> GraphState:
    """Process the query with retrieved documents and tools."""
    chain = prompt | groqLLM_with_tools
    response = chain.invoke({
        "messages": state["messages"],
        "user_query": state['user_query'],
        "retrieved_documents": state['retrieved_documents']
    })
    state["messages"] = state["messages"] + [response]
    return state

tool_node = ToolNode(tools=tools)

workflow = StateGraph(GraphState)
workflow.add_node("input", input_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("processing", processing_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "input")
workflow.add_edge("input", "retrieval")
workflow.add_edge("retrieval", "processing")
workflow.add_conditional_edges("processing", tools_condition)
workflow.add_edge("tools", "processing")
workflow.add_edge("processing", END)

# Add memory
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Chat loop with error handling
config = {"configurable": {"thread_id": "1"}}

# print("Start chatting! Type 'exit' to quit.")
while True:
    try:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        initial_state = {
            "user_query": user_query,
            "retrieved_documents": [],
            "messages": app.get_state(config).values.get("messages", [])  # Load previous messages
        }
        
        result = app.invoke(initial_state, config=config)
        print("Assistant:", result["messages"][-1].content)
    except Exception as e:
        print(f"Error: {str(e)}")
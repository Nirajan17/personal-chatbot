import streamlit as st
import os
from dotenv import load_dotenv
import time
import warnings
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable warnings
warnings.filterwarnings(action="ignore")
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="digitalME Assistant", page_icon="🤖", layout="wide")
st.title("digitalME Assistant")
st.subheader("Your personal AI assistant powered by LangGraph and Groq")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for API keys
with st.sidebar:
    st.header("Configuration")
    
    # API Keys input
    groq_api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", value=os.environ.get("PINECONE_API_KEY", ""), type="password")
    
    if st.button("Save API Keys"):
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        st.success("API keys saved!")

    st.divider()
    st.markdown("### About")
    st.markdown("digitalME is an AI assistant created by Berojgar & company, designed to engage in natural, helpful conversations.")
    st.markdown("It can access documents, read/write files, and query databases to provide enhanced responses.")

# Main application logic
def initialize_app():
    """Initialize the LangGraph application with all components."""
    if not groq_api_key or not pinecone_api_key:
        st.warning("Please provide API keys in the sidebar.")
        return None
    
    # Initialize Groq LLM
    from langchain_groq import ChatGroq
    
    groq_llm = ChatGroq(
        model="deepseek-r1-distill-qwen-32b",
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=512
    )
    
    # Initialize embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize Pinecone
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "personal"
        
        # Create or connect to Pinecone index
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name not in existing_indexes:
            with st.spinner("Creating Pinecone index..."):
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
        
        # Set up retriever
        groq_retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None
    
    # Define tools
    from langchain_core.tools import tool
    
    @tool
    def file_reader(file_path: str) -> str:
        """Read content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
    @tool
    def write_to_file(file_path: str, content: str, append: bool = False) -> str:
        """
        Write or append content to a specified file.
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
    
    # SQL Database tool
    try:
        from langchain_community.utilities import SQLDatabase
        from langchain import hub
        
        db = SQLDatabase.from_uri("sqlite:///transcript.db")
        query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        
        @tool
        def sql_database_tool(user_query: str) -> str:
            """Generate and execute an SQL query based on the user's query."""
            try:
                # Step 1: Generate the SQL query
                prompt = query_prompt_template.invoke(
                    {
                        "dialect": db.dialect,
                        "top_k": 10,
                        "table_info": db.get_table_info(),
                        "input": user_query,
                    }
                )
                structured_groq_llm = groq_llm.with_structured_output(QueryOutput)
                query_result = structured_groq_llm.invoke(prompt)
                sql_query = query_result["query"]
    
                # Step 2: Execute the SQL query
                from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
                execute_query_tool = QuerySQLDatabaseTool(db=db)
                sql_result = execute_query_tool.invoke(sql_query)
    
                return sql_result
            except Exception as e:
                return f"Error executing SQL query: {str(e)}"
    except Exception as e:
        st.warning(f"SQL database initialization error: {str(e)}")
        
        @tool
        def sql_database_tool(user_query: str) -> str:
            """Placeholder for SQL database tool."""
            return "SQL database is not available."
    
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
      - For file-related tasks, use the available tools, execute them fully, and include their results in your response.
      - For database-related queries, use the `sql_database_tool` by passing the exact user query as `user_query`.
    - If a tool returns an error, report it clearly in the response.
    - If `sql_result` is available from a previous database query, incorporate it into your response naturally.
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
        """Initialize messages if not present or add the new message."""
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
            state['retrieved_documents'] = []
        return state
    
    def processing_node(state: GraphState) -> GraphState:
        """Process user query and generate response."""
        chain = prompt | groq_llm_with_tools
        response = chain.invoke(
            {
                "messages": state["messages"],
                "user_query": state['user_query'],
                "retrieved_documents": state['retrieved_documents'],
                "sql_result": state.get("sql_result", "")
            }
        )
        
        # Add response to messages
        state["messages"].append(response)
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
    # workflow.add_edge("processing", END)
    
    # Set up memory for conversation history
    from langgraph.checkpoint.memory import MemorySaver
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
#         st.markdown(message.content)

# Display chat history
for message in st.session_state.messages:
    # Skip displaying raw tool messages
    if hasattr(message, 'type') and message.type == 'tool':
        continue
        
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Get user input
user_query = st.chat_input("Type your message here...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Initialize app if needed
    app = initialize_app()
    
    if app:
        # Display assistant thinking indicator
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("Thinking...")
            
            try:
                # Prepare initial state
                config = {"configurable": {"thread_id": "1"}}
                
                # Convert session state messages to LangChain format
                langchain_messages = []
                for msg in st.session_state.messages[:-1]:  # Exclude the most recent user message
                    if isinstance(msg, HumanMessage):
                        langchain_messages.append(HumanMessage(content=msg.content))
                    else:
                        langchain_messages.append(AIMessage(content=msg.content))
                
                initial_state = {
                    "user_query": user_query,
                    "retrieved_documents": [],
                    "messages": langchain_messages,
                    "sql_result": ""
                }
                
                # Generate response
                result = app.invoke(initial_state, config=config)
                
                # Update the placeholder with the actual response
                assistant_message = result["messages"][-1]
                thinking_placeholder.markdown(assistant_message.content)
                
                # Add assistant message to chat history
                st.session_state.messages.append(AIMessage(content=assistant_message.content))
                
            except Exception as e:
                thinking_placeholder.markdown(f"I encountered an error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=f"I encountered an error: {str(e)}"))
    else:
        # Display error if app initialization failed
        with st.chat_message("assistant"):
            st.markdown("I couldn't initialize properly. Please check your API keys in the sidebar.")
            st.session_state.messages.append(AIMessage(content="I couldn't initialize properly. Please check your API keys in the sidebar."))

# Add file uploader to the sidebar for populating the vector store
with st.sidebar:
    st.divider()
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload documents to the vector store", accept_multiple_files=True, type=["txt", "pdf"])
    
    if uploaded_files and st.button("Process Uploads"):
        with st.spinner("Processing documents..."):
            try:
                # Initialize necessary components
                from langchain_huggingface import HuggingFaceEmbeddings
                from pinecone import Pinecone
                from langchain_pinecone import PineconeVectorStore
                from langchain_community.document_loaders import PyPDFLoader, TextLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Initialize Pinecone
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index("personal")
                
                # Initialize vector store
                vector_store = PineconeVectorStore(index=index, embedding=embeddings)
                
                # Text splitter for chunking documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    # Save the uploaded file temporarily
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and split the document
                    if temp_file_path.endswith(".pdf"):
                        loader = PyPDFLoader(temp_file_path)
                    else:
                        loader = TextLoader(temp_file_path)
                    
                    documents = loader.load()
                    chunks = text_splitter.split_documents(documents)
                    
                    # Add documents to vector store
                    vector_store.add_documents(chunks)
                    
                    # Remove temporary file
                    os.remove(temp_file_path)
                
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

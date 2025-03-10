from langchain_ollama import ChatOllama

deepseek = ChatOllama(
    model="deepseek-r1",
    temperature=0,
)

import time

index_name = "personal" 

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever_deepseek = vector_store.as_retriever(k=3)

from langchain_core.tools import tool

@tool
def marks_reader(self, file_path: str) -> str:
    """Read content from a text file."""
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at path {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
tools = [marks_reader]
deepseek_with_tools = deepseek.bind_tools(tools)

from langchain.prompts import PromptTemplate

template = """
    You are a helpful assistant that can provide information based on both vector search from a database and other tools.

    ### Task:
    You will be provided with a user query. Your goal is to respond to the query by doing the following:
    1. Retrieve relevant information from the vector database (Pinecone) that best matches the query.
    2. If additional context or information is required that can be retrieved from tools (such as reading a PDF), use the appropriate tool.
    3. Combine the information from the database and tools to provide a well-structured and complete response.

    ### User Query:
    {user_query}

    ### Relevant Documents (from Pinecone vector database):
    {retrieved_documents}

    ### Tool Usage:
    If the relevant documents or context are not sufficient to answer the query, use the tools you have. For example, if the user is asking for marks of certain subjects, use the marks_reader tool to extract information from the document.
    If you used any tools, describe how the tool was used in your response.

    ### Answer:
    Your answer should be concise, clear, and comprehensive, using the retrieved documents and any tool-assisted information.
 """

prompt = PromptTemplate(
    input_variables=["user_query", "retrieved_documents"],
    template=template,
)

chain = prompt | deepseek

user_query = input("Your query :: \n")
retrieved_documents = retriever_deepseek.invoke(user_query)

response = chain.invoke({"user_query": user_query, "retrieved_documents": retrieved_documents})
print(response.content)
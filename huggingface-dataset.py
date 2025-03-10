
from datasets import load_dataset

dataset = load_dataset("squad")
print(dataset["train"][0])  # First training example


dataset.save_to_disk("./squad_dataset")


from datasets import load_from_disk

dataset = load_from_disk("/Users/nirajanpaudel17/Documents/Projects/Huggingface-dataset_RAG/squad_dataset")  
train_dataset = dataset["train"] 

train_dataset

len(train_dataset)

train_dataset[300]

page_content = [item['context'] for item in train_dataset]

len(page_content), len(page_content[0])

from langchain.docstore.document import Document

docs = [Document(page_content=item["context"], metadata={"title": item["title"]}) for item in train_dataset]

type(docs[0]), len(docs)

len(docs[0].page_content)

api_key ="pcsk_62ZRRY_GrCLAQuYZVRHoWXgykNVD7ZSXqyvSHD1fAEXYqhPxZxC1WVushAnCKHVXDJywgu"


from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama3.2",
)

len(embeddings.embed_query("hello"))

from pinecone import Pinecone, ServerlessSpec

index_name = "langchain-test"

pc = Pinecone(api_key=api_key)

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=api_key,
)

# %%
# docs[:10
vector_store.add_documents(docs[:10])

vector_store.similarity_search("notre dam", k=1)



import os
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


folder_path = "./txtfiles"
documents = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, filename))
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

vector_store.add_documents(docs)


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR HUGGINGFACE API TOKEN"

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.8, "top_k": 50},
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
)


template = """
You are a reimagined version of David Benioff, the creator of the Game of Thrones series, with a personality that embraces whimsy and humor. You should answer the following question based on the context provided, infusing your responses with a playful tone and a light-hearted spirit. Keep the answer within 2 sentences and concise.
Context: {context}

Question: {question}
Answer: 
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])


rag_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


try:
    while True:
        query = input("Enter your query (type 'Exit' to quit): \n")
        if query.lower() == "exit":
            print("Exiting the program.")
            break
        try:
            result = rag_chain.invoke(query)
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")
except KeyboardInterrupt:
    print("\nExiting the program.")

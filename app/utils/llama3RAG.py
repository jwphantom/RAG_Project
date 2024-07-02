from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_community.document_loaders import PyMuPDFLoader

from langchain_groq import ChatGroq
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import r
from langchain.schema import Document


# Setup for embedding and LLM
def setup_embedding_and_llm():
    # Assuming environment variables are used to configure keys

    groq_api_key = os.getenv("GROQ_API_KEY")

    # llm = ChatGroq(
    #     groq_api_key=groq_api_key,
    #     model_name="llama3-70b-8192",
    #     temperature=0.5,
    # )

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2).bind(logprobs=True)

    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, top_p=0.2)

    embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")

    # embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    return llm, embedding


# Load and split document text
def load_and_split_document(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    chunk_size = 1000
    chunk_overlap = 100

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(data)

    return all_splits


# Fonction pour configurer le retriever
def create_retriever(pdf_path):
    _, embedding = setup_embedding_and_llm()
    all_splits = load_and_split_document(pdf_path)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)
    retriever = vectorstore.as_retriever()
    return retriever


def create_hybrid_retriever(pdf_path):
    _, embedding = setup_embedding_and_llm()
    all_splits = load_and_split_document(pdf_path)
    docs = [
        Document(page_content=split.page_content, metadata=split.metadata)
        for split in all_splits
    ]

    bm25_retriever = BM25Retriever.from_documents(docs)
    faiss_vectorstore = FAISS.from_documents(docs, embedding)
    faiss_retriever = faiss_vectorstore.as_retriever()

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def format_query(question):
    # Start with the basic question format
    formatted_query = f"""Question : {question} \
        Instructions on how to respond: \
        1 - Responses should be in a complete, simple sentence. \
        2 - Your response must be in FRENCH. \
        3 - Do not use phrases like **information provided, text etc.**, respond as an agent would. \
        4 - If a request is out of the context of the document, respond with a phrase like you do not understand and that the request is out of context, without mentioning that you searched in the text. \
        5 - If you cannot find a similarity to the question asked in the documents, respond that it is out of context. \
        6 - If you have difficulties finding information in the text, do not say that you did not find it but that it is out of context. \
    """

    return formatted_query

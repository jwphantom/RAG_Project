from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_groq import ChatGroq
import os


# Setup for embedding and LLM
def setup_embedding_and_llm():
    # Assuming environment variables are used to configure keys

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2,
    )
    embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")

    return llm, embedding


# Load and split document text
def load_and_split_document(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # all_splits = text_splitter.split_documents(data)
    return data


def format_query(question):
    # Format the question as required
    return f""" Question : {question} \
        Instruction sur comment répondre : \
        1 - Réponses par une phrase complète, simple et en français \
        2 - NE pas mettre les expressions du genre **informations fournies**, réponds comme un agent
        3 - Si une des demandes est hors contexte du document, répondre en disant que c'est hors contexte  
        """

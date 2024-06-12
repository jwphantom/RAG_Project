from langchain_community.vectorstores import Chroma
from app.utils.llama3RAG import (
    format_query,
    load_and_split_document,
    setup_embedding_and_llm,
)
from langchain.chains import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings


import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

modelGemini = genai.GenerativeModel("gemini-pro")


def correct_spelling(text):
    # Assuming modelGemini is already initialized and available
    corrected_text = modelGemini.generate_content(
        f"Corrige les fautes de grammaire et d’orthographe de la phrase dans les guillemets, fournis juste la phrase, pas autre chose : << {text} >> "
    )
    return corrected_text.text


def generate_prompt(input, pdf_path):
    corrected_input = correct_spelling(input)

    llm, embedding = setup_embedding_and_llm()

    all_splits = load_and_split_document(pdf_path)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

    chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever(), verbose=True
    )
    query = format_query(corrected_input)

    responses = chain.invoke({"query": query})

    return {"Responses": responses["result"]}

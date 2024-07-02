from app.utils.llama3RAG import (
    format_query,
    load_and_split_document,
    setup_embedding_and_llm,
)
from langchain.chains import RetrievalQA

import google.generativeai as genai
import os

from langchain.prompts import PromptTemplate

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

modelGemini = genai.GenerativeModel("gemini-pro")


def generate_prompt(input, pdf_path, user, memory):

    llm, embedding = setup_embedding_and_llm()

    all_splits = load_and_split_document(pdf_path)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

    template = """
            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
            Instructions on how to respond: \
            - For questions with explicit answers in the passage, respond in full without rephrasing. \
            - For questions without explicit answers in the passage, respond by combining the information gathered in the document. \
            - The sentences must be coherent and without mistakes. \
            - Responses must be in FRENCH. \
            - Do not use phrases like "According to the document," "According to the passage," "The provided information," "In the text," etc. Instead, respond as if you were a human with this knowledge acquired since birth. \
            - If the question is out of context or unrelated, simply say: "Je suis désolé mais cela va au-delà de mes capacités." \
            - If you have difficulties finding the context, do not mention "According to the document," "According to the passage," "The provided information," "In the text," etc. Simply say: "Je suis désolé mais cela va au-delà de mes capacités." \
            - Use the chat history between <hs></hs> to provide consistent conversation tracking.
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        {question}
        Answer:
        """

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    responses = chain.invoke({"query": input})

    # print(responses)

    return {"Responses": responses["result"]}

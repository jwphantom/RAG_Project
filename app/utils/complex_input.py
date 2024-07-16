from langchain_community.vectorstores import Chroma
from app.utils.llama3RAG import (
    create_retriever,
    load_and_split_document,
    setup_embedding_and_llm,
    create_hybrid_retriever,
)
from langchain.chains import RetrievalQA

import google.generativeai as genai
import os

from langchain.prompts import PromptTemplate

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

modelGemini = genai.GenerativeModel("gemini-pro")


def generate_prompt(input, pdf_path, user, history):

    llm, _ = setup_embedding_and_llm()

    retriever = create_hybrid_retriever(pdf_path)
    docs = retriever.get_relevant_documents(input)
    context = "\n".join([doc.page_content for doc in docs])

    # Template pour le prompt
    template = """
            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:\n
            Instructions on how to respond: \n
            - For questions with explicit answers in the passage, respond in full without rephrasing. \n
            - For questions without explicit answers in the passage, respond by combining the information gathered in the document. \n
            - The sentences must be coherent and without mistakes. \n
            - Responses must be in FRENCH. \n
            - Do not use phrases like "According to the document," "According to the passage," "The provided information," "In the text," etc. Instead, respond as if you were a human with this knowledge acquired since birth. \n
            - If the question is out of context or unrelated, simply say: "Je suis désolé mais cela va au-delà de mes capacités." \n
            - If you have difficulties finding the context, do not mention "According to the document," "According to the passage," "The provided information," "In the text," etc. Simply say: "Je suis désolé mais cela va au-delà de mes capacités." \n
            - Use the chat history between <hs></hs> to provide consistent conversation tracking.\n
            - Use the history to remember the exchange \n
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        User : Merci.
        Bot: La formation en licence de pilote de ligne est conçue pour préparer les étudiants à une carrière professionnelle dans l'aviation commerciale. Ce programme combine des cours théoriques approfondis avec une formation pratique intensive en vol. Les étudiants apprendront les principes de l'aérodynamique, la navigation aérienne, les systèmes d'avion, la météorologie, et les réglementations aériennes. La formation inclut également des heures de vol sous la supervision d'instructeurs qualifiés, permettant aux étudiants d'acquérir les compétences nécessaires pour piloter différents types d'aéronefs.
        User : Bonjour monsieur, Je voudrais les informations sur votre formation de pilote de ligne
        </hs>
        ------
        {question}
        Réponse :
        """

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    # Générer le prompt formaté
    formatted_prompt = prompt.format(history=history, context=context, question=input)

    responses = llm.invoke(formatted_prompt)

    return responses

import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import math
import time

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

modelGemini = genai.GenerativeModel("gemini-1.5-pro")
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, top_p=0.2)

phrases_test = [
    "Le chat mange une souris.",
    "Je vais au cinéma ce soir.",
    "L'avion décolle à 10h30.",
]


def fraction_to_float(fraction_str):
    try:
        return float(fraction_str)
    except ValueError:
        num, denom = map(int, fraction_str.split("/"))
        return num / denom


def calculer_perplexite(phrases):
    log_probs_totales = 0
    total_tokens = 0

    for phrase in phrases:
        ids = llm.get_token_ids(phrase)
        total_tokens += len(ids)

        log_probs_phrase = 0
        for i in range(len(ids)):
            contexte = ids[:i]
            token_actuel = ids[i]

            result = llm.invoke(
                f"Quelle est la probabilité du token {token_actuel} après la séquence {contexte}? Répondez uniquement par un nombre ou une fraction."
            )

            proba = fraction_to_float(result.content)

            print(f"Token: {token_actuel}, Probabilité: {proba}")

            log_probs_phrase += math.log(proba) if proba > 0 else float("-inf")

            # Ajouter un délai de 2 secondes
            time.sleep(4)

        log_probs_totales += log_probs_phrase

    moyenne_log_probs = log_probs_totales / total_tokens
    perplexite = math.exp(-moyenne_log_probs)

    return perplexite


perplexite = calculer_perplexite(phrases_test)
print(f"La perplexité du modèle est : {perplexite}")

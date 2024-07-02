import json
from langchain_openai import ChatOpenAI
import numpy as np
import math
import time
import csv
from fastapi import APIRouter, HTTPException
from typing import List
from dotenv import load_dotenv
import nltk
from datetime import datetime

nltk.download("punkt")
from nltk.tokenize import word_tokenize

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.memory import ConversationBufferMemory

from app.schema.question_eval import Question as SchemaQuestionEval
from app.schema.question_eval import ResponseEvaluation as SchemaResponseEval
from app.utils.complex_input import (
    generate_prompt,
)  # Ensure this is correctly implemented

load_dotenv(".env")

router = APIRouter()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2).bind(logprobs=True)


def init_embedding():
    return GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def fraction_to_float(fraction_str):
    try:
        return float(fraction_str)
    except ValueError:
        num, denom = map(int, fraction_str.split("/"))
        return num / denom


def calculate_ttr(texts: List[str], question_idx: int) -> float:
    print(
        f"Calcul TTR question : {question_idx + 1} .... veuillez patienter ça calcule"
    )
    all_tokens = []
    unique_tokens = set()

    for text in texts:
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)
        unique_tokens.update(tokens)

    if len(all_tokens) == 0:
        ttr = 0
    else:
        ttr = len(unique_tokens) / len(all_tokens)

    print(f"Result TTR question : {question_idx + 1} .... {ttr}")
    return ttr


def calculate_metrics_from_log_probs(log_probs: List[float], question_idx: int):

    print(
        f"Calcul Entropie et Perplexité question : {question_idx + 1} .... veuillez patienter ça calcule"
    )

    total_log_prob = sum(log_probs)
    total_tokens = len(log_probs)

    moyenne_log_probs = total_log_prob / total_tokens
    entropy = -moyenne_log_probs
    perplexity = math.exp(entropy)

    print(f"Result Entropie question : {question_idx + 1} .... {entropy}")
    print(f"Result Perplexité question : {question_idx + 1} .... {perplexity}")

    return {"entropy": entropy, "perplexity": perplexity}


def evaluate_response(
    generated_response, reference_responses, embedding, question_idx: int
):
    print(
        f"Calcul Similarité question : {question_idx + 1} .... veuillez patienter ça calcule"
    )
    gen_vec = embedding.embed_query(generated_response)
    similarities = []
    for response in reference_responses:
        ref_vec = embedding.embed_query(response)
        similarity = cosine_similarity(gen_vec, ref_vec)
        similarities.append(similarity)

    print(f"Result Similarité question : {question_idx + 1} .... {similarities[0]}")

    return similarities


@router.post("/evaluate-responses")
async def evaluate_responses():
    results = {}
    embedding = init_embedding()

    with open("data.json", "r") as f:
        questions = json.load(f)

    # Initialiser le fichier JSON avec une structure de base
    with open("results.json", "w") as f:
        json.dump({category: [] for category in questions.keys()}, f, indent=4)

    for category, questions_list in questions.items():
        start_time_category = time.time()
        perplexity_total = 0

        for idx, question in enumerate(questions_list):
            memory = ConversationBufferMemory(
                memory_key="history", input_key="question"
            )

            schema_question = SchemaQuestionEval(
                prompt=question["prompt"], answer_correct=question["answer_correct"]
            )

            start_time = time.time()
            generated_response = generate_prompt(
                schema_question.prompt, "QR3.pdf", "user", memory
            )
            end_time = time.time()
            generation_time = end_time - start_time

            similarity = evaluate_response(
                generated_response.content,
                [schema_question.answer_correct],
                embedding,
                idx,
            )

            response_metadata = generated_response.response_metadata["logprobs"][
                "content"
            ]
            log_probs = [token_info["logprob"] for token_info in response_metadata]

            metrics = calculate_metrics_from_log_probs(log_probs, idx)

            ttr = calculate_ttr([generated_response.content], idx)

            evaluation = {
                "prompt": schema_question.prompt,
                "answer_correct": schema_question.answer_correct,
                "answer_generated": generated_response.content,
                "ttr": ttr,
                "cosine_similarity": similarity[0],
                "entropy": metrics["entropy"],
                "perplexity": metrics["perplexity"],
                "generation_time": generation_time,
            }

            perplexity_total += metrics["perplexity"]

            # Écrire le résultat individuel dans le fichier JSON
            with open("results.json", "r+") as f:
                data = json.load(f)
                data[category].append(evaluation)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()

            time.sleep(3)

        category_perplexity = perplexity_total / len(questions_list)

        # Ajouter la perplexité de la catégorie
        with open("results.json", "r+") as f:
            data = json.load(f)
            data[category].append({"category_perplexity": category_perplexity})
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.truncate()

        end_time_category = time.time()
        category_duration = end_time_category - start_time_category
        print(f"Category '{category}' processed in {category_duration:.2f} seconds")

        results[category] = data[category]

    return results


@router.post("/read-and-analyze-results")
def read_and_analyze_results():
    analysis = []
    with open("results.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            perplexity = float(row["perplexity"])
            if not math.isfinite(perplexity):
                perplexity = "undefined"

            analysis.append(
                {
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "correct_answer": row["answer_correct"],
                    "generated_answer": row["answer_generated"],
                    "cosine_similarity": float(row["cosine_similarity"]),
                    "entropy": (
                        float(row["entropy"])
                        if math.isfinite(float(row["entropy"]))
                        else "undefined"
                    ),
                    "perplexity": (
                        float(row["perplexity"])
                        if math.isfinite(float(row["perplexity"]))
                        else "undefined"
                    ),
                    "generation_time": float(row["generation_time"]),
                }
            )

    return analysis

# build a schema using pydantic
from pydantic import BaseModel


class Question(BaseModel):
    prompt: str
    answer_correct: str


class ResponseEvaluation(BaseModel):
    prompt: str
    answer_correct: str
    answer_generated: str
    cosinus_similarity: float

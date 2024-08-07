# build a schema using pydantic
from pydantic import BaseModel


class Question(BaseModel):
    prompt: str
    user: str

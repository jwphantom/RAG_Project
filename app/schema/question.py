# build a schema using pydantic
from pydantic import BaseModel


class Question(BaseModel):
    prompt: str

    class Config:
        orm_mode = True

# build a schema using pydantic
from pydantic import BaseModel


class Conversation(BaseModel):
    prompt: str
    user: str
    response: str

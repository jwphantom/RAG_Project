from fastapi import APIRouter, FastAPI

from app.schema.question import Question as SchemaQuestion
from app.utils.complex_input import generate_prompt


from dotenv import load_dotenv

load_dotenv(".env")


router = APIRouter()


@router.post("/generate-prompt")
async def chat(question: SchemaQuestion):
    path_pdf = "QR2.pdf"
    response = generate_prompt(question.prompt, path_pdf)

    return response

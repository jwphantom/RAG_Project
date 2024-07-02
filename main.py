from fastapi import FastAPI
from app.api import chat, evaluation

from fastapi.middleware.cors import CORSMiddleware


import os
from dotenv import load_dotenv

load_dotenv()

origins = [
    "http://localhost:3000",
]


app = FastAPI()

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

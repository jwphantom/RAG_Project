import json
import os
from fastapi import APIRouter, HTTPException
from app.schema.question import Question as SchemaQuestion
from app.schema.conversation import Conversation as SchemaConversation

from app.utils.complex_input import generate_prompt


from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory


load_dotenv(".env")

router = APIRouter()


# Assuming the JSON file is named `conversations.json` and exists in the root directory
def save_conversation(conversation: SchemaConversation):
    user_folder = "user_conversations"
    user_file = f"{user_folder}/{conversation.user}.json"

    os.makedirs(user_folder, exist_ok=True)  # Ensure the directory exists

    try:
        if os.path.exists(user_file):
            with open(user_file, "r+") as file:
                data = json.load(file)
                data.append(conversation.dict())
                file.seek(0)
                json.dump(
                    data, file, ensure_ascii=False, indent=4
                )  # Use ensure_ascii=False here
        else:
            with open(user_file, "w") as file:
                json.dump(
                    [conversation.dict()], file, ensure_ascii=False, indent=4
                )  # And here
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_formatted_conversation(user_id: str) -> str:
    user_folder = "user_conversations"
    user_file = f"{user_folder}/{user_id}.json"

    try:
        if os.path.exists(user_file):
            with open(user_file, "r") as file:
                data = json.load(file)
                if not data:  # Check if the data list is empty
                    return None
                conversation_str = ""
                for item in data:
                    conversation_str += (
                        f"user: {item['prompt']}\nbot: {item['response']}\n"
                    )
                return conversation_str
        else:
            return None  # Return None if the file does not exist
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


memory = ConversationBufferMemory(memory_key="history", input_key="question")


@router.post("/generate-prompt")
async def chat(question: SchemaQuestion):
    path_pdf = "QR3.pdf"

    history = get_formatted_conversation(question.user)

    response = generate_prompt(question.prompt, path_pdf, question.user, history)

    # # Create a conversation object and save it
    # conversation = SchemaConversation(
    #     prompt=question.prompt, user=question.user, response=response.content
    # )
    # save_conversation(conversation)

    return response

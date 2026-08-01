from fastapi import APIRouter

from src.models.chat_models import UserMessage
from src.services.chat_service import process_user_message as _process_user_message


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/user-msg")
async def user_message_handler(data: UserMessage):
    return await _process_user_message(data)

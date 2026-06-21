from fastapi import APIRouter

from src.models.chat_models import UserMessage, SymbolExtractionResponse
from src.services.chat_service import process_user_message as _process_user_message


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/user-msg", response_model=SymbolExtractionResponse)
async def user_message_handler(data: UserMessage) -> SymbolExtractionResponse:
    return await _process_user_message(data)

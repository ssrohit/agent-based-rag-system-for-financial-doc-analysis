from fastapi import APIRouter

from src.models.chat_models import UserMessage
from src.services.agent_service import run_agent as _run_agent


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/ask")
async def agent_ask_handler(data: UserMessage):
    return await _run_agent(data)

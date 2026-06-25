"""
Chat Service to process user messages
"""
import logging

from src.models.chat_models import UserMessage
from langfuse import get_client, observe

logger = logging.getLogger(__name__)
langfuse = get_client()

@observe()
async def process_user_message(data: UserMessage):
    pass
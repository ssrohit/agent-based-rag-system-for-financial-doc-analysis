from fastapi import APIRouter

from src.models.chat_models import UserMessage
from src.services.ingestion_service import ingest_data


router = APIRouter(prefix="/ingest",tags=['ingest'])

@router.post('')
async def ingest(userQuery: UserMessage):
    return await ingest_data(userQuery)


from typing import Union,List

from pydantic import BaseModel, ConfigDict, Field
from src.models.chat_models import UserMessage

class UserQuery(UserMessage):
    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, use_enum_values=True
    )

class SymbolExtraction(BaseModel):
    symbol: Union[str, List[str]] = Field(
        description="US stock symbol or list of US stock symbols"
    )
    from_date: str = Field(
        description="From date in ISO 8601 format (YYYY-MM-DDThh:mm:ss)"
    )
    to_date: str = Field(
        description="To date in ISO 8601 format (YYYY-MM-DDThh:mm:ss)"
    )
    
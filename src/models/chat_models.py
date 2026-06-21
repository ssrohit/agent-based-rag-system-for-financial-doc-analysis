from typing import List, Union

from pydantic import BaseModel, Field, ConfigDict


class UserMessage(BaseModel):
    user_message: str = Field(alias="userMessage", description="Message sent by user")

    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, use_enum_values=True
    )


class SymbolExtractionResponse(BaseModel):
    symbol: Union[str, List[str]] = Field(
        description="US stock symbol or list of US stock symbols"
    )
    from_date: str = Field(
        description="From date in ISO 8601 format (YYYY-MM-DDThh:mm:ss)"
    )
    to_date: str = Field(
        description="To date in ISO 8601 format (YYYY-MM-DDThh:mm:ss)"
    )

from typing import List, Union

from pydantic import BaseModel, Field, ConfigDict


class UserMessage(BaseModel):
    user_message: str = Field(alias="userMessage", description="Message sent by user")

    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, use_enum_values=True
    )


from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class UserMessage(BaseModel):
    user_message: str = Field(alias="userMessage", description="Message sent by user")

    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, use_enum_values=True
    )


class SourceReference(BaseModel):
    """Citation metadata for a retrieved chunk, taken directly from the vector store
    (never LLM-generated) so citations can't be hallucinated."""

    company_name: Optional[str] = None
    doc_type: Optional[str] = None
    filing_date: Optional[str] = None
    cik: Optional[str] = None


class FinancialAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the user's question")
    key_metrics: List[str] = Field(
        default_factory=list, description="Relevant figures/metrics cited from the context"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Caveats, limitations, or missing information relevant to the answer",
    )


class ChatResponse(BaseModel):
    answer: str
    key_metrics: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    sources: List[SourceReference] = Field(default_factory=list)


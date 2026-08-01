import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get path to config.py directory
CONFIG_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    CHAT_MODEL: str = Field(description="LLM model to be used", examples=["gpt-4.1"])
    GOOGLE_API_KEY: str = Field(description="API Key of the model")
    CHROMA_PERSIST_PATH: str = Field(description="Chroma DB persist path")
    VECTOR_GENERATION_MODEL: str = Field(description="Model to generate embeddings")
    LANGFUSE_SECRET_KEY: str = Field()
    LANGFUSE_PUBLIC_KEY: str = Field()
    LANGFUSE_BASE_URL: str = Field()
    HF_TOKEN: str = Field()

    model_config = SettingsConfigDict(env_file=CONFIG_DIR / "configs" / ".env")


# Single shared instance — import `settings` everywhere instead of calling Settings()
settings: Settings = Settings()

# Populate OS environment variables so that external libraries (like Langfuse) can read them
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_BASE_URL
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
os.environ["HF_TOKEN"] = settings.HF_TOKEN
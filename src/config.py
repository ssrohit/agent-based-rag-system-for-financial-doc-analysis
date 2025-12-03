from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    CHAT_MODEL: str = Field(description="LLM model to be used", example="gpt-4.1")
    MODEL_API_KEY: str = Field(description="API Key of the model")
    
    model_config = SettingsConfigDict(env_file=".env")
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import json  # noqa: F401
import logging
import uvicorn
from fastapi import FastAPI

from langchain_core.prompts import PromptTemplate  # noqa: F401
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401

from src.config import Settings  # noqa: F401
from src.routes.chat import router as chat_router
from src.routes.ingest import router as ingest_router

app = FastAPI()
app.include_router(chat_router)
app.include_router(ingest_router)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # config = Settings()
    # model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config.GOOGLE_API_KEY)
    # systemMessage = SystemMessage(content="Extract the US stock symbol and the time range from the provided query and give the response as an object with extracted symbol, from and to")
    # humanMessage = HumanMessage(content="Analyze the apple stock in the time range of 2024-2026")
    # PromptTemplate.from_template()
    # messages = [systemMessage, humanMessage]
    # response = model.invoke(messages)
    # print(response)
    # print(json.loads(response.content))

    # Run the uvicorn server programmatically
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()


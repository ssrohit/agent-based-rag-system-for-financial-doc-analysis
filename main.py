from src.config import Settings
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    config = Settings()
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config.GOOGLE_API_KEY)
    systemMessage = SystemMessage("You are a helpful assistant")
    humanMessage = HumanMessage("Give me a simple python logging example")
    messages = [systemMessage, humanMessage]
    response = model.invoke(messages)
    print(response)



if __name__ == "__main__":
    main()

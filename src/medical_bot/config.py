import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    INDEX_NAME = "medical-chatbot"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    AWS_REGION = "us-east-1"
    CHAT_MODEL = "gpt-4o"

    @classmethod
    def validate(cls):
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set is the environment variables.")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set is the environment variables.")

# Auto-validate/expose keys for legacy compatibility if needed, 
# but better to use the class attributes directly.
os.environ["PINECONE_API_KEY"] = AppConfig.PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = AppConfig.OPENAI_API_KEY or ""

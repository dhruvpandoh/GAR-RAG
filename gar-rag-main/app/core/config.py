## app/core/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://localhost/gar_rag_search"
    LLAMA_MODEL_PATH: str = "./models/llama-2-3.2b.gguf"
    
    class Config:
        env_file = ".env"

settings = Settings()

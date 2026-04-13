import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = "ollama"

OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "llama3.1:8b-instruct-q4_K_M",
    "temperature": 0.1,
    "format": "json",
}

GEMINI_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "model": "google/gemini-flash-2.0",
    "temperature": 0.1,
}

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from mas_validation.config import LLM_BACKEND, OLLAMA_CONFIG, GEMINI_CONFIG


def get_llm():
    if LLM_BACKEND == "ollama":
        return ChatOllama(
            base_url=OLLAMA_CONFIG["base_url"],
            model=OLLAMA_CONFIG["model"],
            temperature=OLLAMA_CONFIG["temperature"],
            format=OLLAMA_CONFIG["format"],
        )
    elif LLM_BACKEND == "gemini":
        return ChatOpenAI(
            base_url=GEMINI_CONFIG["base_url"],
            api_key=GEMINI_CONFIG["api_key"],
            model=GEMINI_CONFIG["model"],
            temperature=GEMINI_CONFIG["temperature"],
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")

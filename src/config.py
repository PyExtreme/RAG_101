"""Configuration module for semantic search engine."""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # Ollama Configuration
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    
    # Search Configuration
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Model parameters
    MAX_EMBEDDING_DIM: int = 768  # nomic-embed-text output dimension
    

def get_config() -> Config:
    """Get application configuration."""
    return Config()

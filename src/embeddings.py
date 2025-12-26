"""Embeddings module using Ollama for local embedding generation."""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import requests
from src.config import Config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OllamaEmbeddings(EmbeddingProvider):
    """
    Ollama-based embeddings using local models.
    
    Why Ollama + nomic-embed-text:
    - FREE: No API costs, completely local
    - FAST: Runs on consumer hardware
    - PRIVATE: No data leaves your machine
    - GOOD: nomic-embed-text is surprisingly capable (768-dim)
    
    Trade-offs vs OpenAI:
    - OpenAI: Better quality, slower to run locally, costs money
    - Ollama: Good quality, instant, free, but requires GPU/CPU for generation
    
    How it works:
    1. Text is sent to local Ollama server
    2. Model encodes text to 768-dimensional vector
    3. Vector captures semantic meaning
    """
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama embeddings.
        
        Args:
            model: Model name (nomic-embed-text is recommended)
            base_url: URL of Ollama server
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embed_endpoint = f"{self.base_url}/api/embed"
        
        # Check if Ollama is running
        try:
            self._check_connection()
        except Exception as e:
            print(f"Warning: Could not connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running. You can start it with: ollama serve")
            print(f"Then pull the model: ollama pull {model}")
            raise
    
    def _check_connection(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return [0.0] * 768  # Return zero vector for empty text
        
        try:
            response = requests.post(
                self.embed_endpoint,
                json={
                    "model": self.model,
                    "input": text.strip()
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            else:
                raise ValueError("No embeddings returned from Ollama")
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Start Ollama with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        # For Ollama, we need to make individual requests per text
        # (API doesn't support batch in standard way)
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Could not embed text: {e}")
                embeddings.append([0.0] * 768)  # Zero vector on error
        
        return embeddings


class EmbeddingFactory:
    """Factory for creating embedding provider instances."""
    
    @staticmethod
    def create(provider: str = "ollama", **kwargs) -> EmbeddingProvider:
        """
        Create an embedding provider.
        
        Args:
            provider: "ollama" (default)
            **kwargs: Provider-specific arguments
        
        Returns:
            EmbeddingProvider instance
        """
        if provider.lower() == "ollama":
            return OllamaEmbeddings(
                model=kwargs.get("model", Config.OLLAMA_MODEL),
                base_url=kwargs.get("base_url", Config.OLLAMA_BASE_URL)
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

"""Chunking strategies for text segmentation."""
from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    chunk_id: str
    source_doc: str
    page_number: int = 0
    chunk_index: int = 0


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, source_doc: str, page_number: int = 0) -> List[TextChunk]:
        """Break text into chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking strategy.
    
    Splits text into chunks of fixed size with optional overlap.
    
    Why this works:
    - Ensures consistent context windows for embeddings
    - Simple and predictable
    - Works well for uniform documents
    
    Trade-offs:
    - May cut sentences in the middle
    - Overlap increases storage but improves recall
    - Not semantically aware (might split related concepts)
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
    
    def chunk(self, text: str, source_doc: str, page_number: int = 0) -> List[TextChunk]:
        """Break text into fixed-size chunks with overlap."""
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 20:
                continue
            
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{source_doc}_chunk_{len(chunks)}",
                source_doc=source_doc,
                page_number=page_number,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking strategy (sentence-aware).
    
    Tries to keep sentences together by breaking at sentence boundaries.
    
    Why this works:
    - Preserves semantic meaning
    - Sentences are natural units of information
    - Better for retrieval relevance
    
    Trade-offs:
    - Variable chunk sizes (might exceed max context)
    - Requires sentence detection (nltk/spacy)
    - Slower than fixed-size chunking
    """
    
    def __init__(self, target_size: int = 500, overlap: int = 100):
        """
        Initialize semantic chunker.
        
        Args:
            target_size: Target number of characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.target_size = target_size
        self.overlap = overlap
    
    def chunk(self, text: str, source_doc: str, page_number: int = 0) -> List[TextChunk]:
        """Break text into semantic chunks (sentence-aware)."""
        # Simple sentence splitting on . ! ? followed by space
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed target size
            if len(current_chunk) + len(sentence) > self.target_size and current_chunk:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        # Create TextChunk objects with overlap
        text_chunks = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 20:
                continue
            
            text_chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{source_doc}_chunk_{len(text_chunks)}",
                source_doc=source_doc,
                page_number=page_number,
                chunk_index=len(text_chunks)
            )
            text_chunks.append(text_chunk)
        
        return text_chunks


class ChunkerFactory:
    """Factory for creating chunker instances."""
    
    _strategies = {
        "fixed": FixedSizeChunker,
        "semantic": SemanticChunker,
    }
    
    @classmethod
    def create(cls, strategy: str = "fixed", **kwargs) -> ChunkingStrategy:
        """Create a chunking strategy."""
        if strategy not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(cls._strategies.keys())}")
        
        return cls._strategies[strategy](**kwargs)

"""Similarity metrics for semantic search."""
import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod


class SimilarityMetric(ABC):
    """Abstract base class for similarity metrics."""
    
    @abstractmethod
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate similarity between two vectors."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass


class CosineSimilarity(SimilarityMetric):
    """
    Cosine similarity metric.
    
    WHY COSINE?
    - Measures angle between vectors, not magnitude
    - Invariant to vector scale (works well for embeddings)
    - Range: -1 to 1 (typically 0 to 1 for embeddings)
    - Most popular for semantic search
    
    INTERPRETATION:
    - 1.0 = identical direction (same meaning)
    - 0.5 = moderately similar
    - 0.0 = orthogonal (no similarity)
    - -1.0 = opposite direction (rare with embeddings)
    
    FORMULA: cos(θ) = (A·B) / (||A|| * ||B||)
    """
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        
        # Handle zero vectors
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return float(dot_product / (norm1 * norm2))
    
    def name(self) -> str:
        return "cosine"


class DotProduct(SimilarityMetric):
    """
    Dot product metric.
    
    WHY DOT PRODUCT?
    - Faster than cosine (no normalization)
    - Good when embeddings are already normalized
    - Raw measure of alignment
    
    Trade-off:
    - Magnitude-dependent (larger vectors score higher)
    - Less intuitive interpretation
    - Use if you need speed and embeddings are normalized
    
    FORMULA: A·B = Σ(a_i * b_i)
    """
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product between two vectors."""
        return float(np.dot(np.array(vec1), np.array(vec2)))
    
    def name(self) -> str:
        return "dot_product"


class EuclideanDistance(SimilarityMetric):
    """
    Euclidean distance metric (L2 distance).
    
    WHY EUCLIDEAN?
    - Geometric distance in vector space
    - Intuitive: "how far apart are the vectors?"
    - Can handle variable magnitude vectors
    
    Trade-off:
    - Slower than cosine
    - Less suitable for high-dimensional spaces (curse of dimensionality)
    - Need to invert: similarity = 1 / (1 + distance)
    
    FORMULA: d = √(Σ(a_i - b_i)²)
    """
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance (converted to similarity)."""
        distance = np.linalg.norm(np.array(vec1) - np.array(vec2))
        # Convert distance to similarity (0 to 1)
        return 1.0 / (1.0 + distance)
    
    def name(self) -> str:
        return "euclidean"


class SimilarityMetricFactory:
    """Factory for creating similarity metrics."""
    
    _metrics = {
        "cosine": CosineSimilarity,
        "dot_product": DotProduct,
        "euclidean": EuclideanDistance,
    }
    
    @classmethod
    def create(cls, metric: str = "cosine") -> SimilarityMetric:
        """Create a similarity metric."""
        if metric not in cls._metrics:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(cls._metrics.keys())}")
        
        return cls._metrics[metric]()
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """List available metrics."""
        return list(cls._metrics.keys())

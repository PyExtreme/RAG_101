"""Vector database integration using ChromaDB."""
from typing import List, Dict, Tuple, Optional
import chromadb
from pathlib import Path
import json
from datetime import datetime
from src.chunking import TextChunk
from src.similarity import SimilarityMetric, SimilarityMetricFactory


class VectorStore:
    """
    ChromaDB-based vector store for semantic search.
    
    WHY CHROMADB?
    - Lightweight vector database (no separate server needed)
    - Automatic embedding storage
    - Built-in similarity search
    - Persistent storage to disk
    - Perfect for learning and production small-to-medium projects
    
    What ChromaDB does under the hood:
    1. Stores embeddings in optimized format
    2. Builds indexes for fast similarity search
    3. Handles metadata filtering
    4. Provides collection-based organization
    5. Syncs to disk for persistence
    
    How embeddings enable search:
    - Each document chunk becomes a vector in N-dimensional space
    - Query also becomes a vector in same space
    - "Most similar" = vectors closest in this space
    - ChromaDB uses efficient search algorithms (approximate nearest neighbor)
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        similarity_metric: str = "cosine"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Where to store ChromaDB files
            collection_name: Name of the collection
            similarity_metric: Similarity metric to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.similarity_metric = SimilarityMetricFactory.create(similarity_metric)
        
        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Create ChromaDB client with persistent storage (new API)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine for HNSW indexing
        )
        
        # Metadata tracking
        self.metadata_file = Path(self.persist_directory) / "chunks_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from file if it exists."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.chunks_metadata = json.load(f)
        else:
            self.chunks_metadata = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(self.chunks_metadata, f, indent=2)
    
    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings to the store.
        
        Args:
            chunks: List of TextChunk objects
            embeddings: List of corresponding embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source_doc": chunk.source_doc,
                "chunk_index": str(chunk.chunk_index),
                "page_number": str(chunk.page_number),
                "added_at": datetime.now().isoformat(),
            }
            for chunk in chunks
        ]
        
        # Add to ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        # Update metadata tracking
        for chunk in chunks:
            self.chunks_metadata[chunk.chunk_id] = {
                "text": chunk.text,
                "source_doc": chunk.source_doc,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
            }
        
        self._save_metadata()
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Tuple[str, float, str, Dict]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            List of tuples: (chunk_id, similarity_score, text, metadata)
        """
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        
        retrieved = []
        for i, chunk_id in enumerate(results["ids"][0]):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            
            # ChromaDB returns distances, convert to similarity
            # For cosine in HNSW: distance = 1 - similarity
            distance = results["distances"][0][i]
            similarity = 1 - distance
            
            retrieved.append((chunk_id, similarity, text, metadata))
        
        return retrieved
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "metadata_file": str(self.metadata_file),
            "persist_directory": self.persist_directory,
        }
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.chunks_metadata = {}
        self._save_metadata()
        print(f"Cleared collection: {self.collection_name}")
    
    def delete_by_source(self, source_doc: str) -> None:
        """Delete all chunks from a specific source document."""
        to_delete = [
            chunk_id for chunk_id, metadata in self.chunks_metadata.items()
            if metadata.get("source_doc") == source_doc
        ]
        
        if to_delete:
            for chunk_id in to_delete:
                del self.chunks_metadata[chunk_id]
            self._save_metadata()
            # Note: ChromaDB delete by ID
            for chunk_id in to_delete:
                try:
                    self.collection.delete(ids=[chunk_id])
                except:
                    pass
            print(f"Deleted {len(to_delete)} chunks from {source_doc}")

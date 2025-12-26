"""Main semantic search engine."""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from src.config import Config
from src.ingestion import DocumentIngester, Document
from src.chunking import ChunkerFactory, TextChunk
from src.embeddings import EmbeddingFactory, EmbeddingProvider
from src.vector_store import VectorStore
from src.similarity import SimilarityMetricFactory


class SemanticSearchEngine:
    """
    Complete semantic search engine.
    
    Pipeline:
    1. Ingest documents (PDF, TXT, MD)
    2. Chunk text intelligently
    3. Generate embeddings for chunks
    4. Store in vector database
    5. Accept queries and retrieve most similar chunks
    """
    
    def __init__(
        self,
        persist_dir: str = Config.CHROMA_DB_PATH,
        embedding_provider: str = "ollama",
        chunking_strategy: str = "fixed",
        similarity_metric: str = "cosine",
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
        top_k: int = Config.TOP_K,
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            persist_dir: ChromaDB persistence directory
            embedding_provider: "ollama"
            chunking_strategy: "fixed" or "semantic"
            similarity_metric: "cosine", "dot_product", or "euclidean"
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            top_k: Number of results to retrieve
        """
        self.config = Config()
        self.top_k = top_k
        
        # Initialize components
        self.embeddings = EmbeddingFactory.create(provider=embedding_provider)
        self.chunker = ChunkerFactory.create(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap if chunking_strategy == "fixed" else 0
        )
        self.vector_store = VectorStore(
            persist_directory=persist_dir,
            similarity_metric=similarity_metric
        )
        self.similarity_metric = SimilarityMetricFactory.create(similarity_metric)
    
    def index_documents(self, directory_path: str, verbose: bool = True) -> Dict:
        """
        Index all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            verbose: Print progress
        
        Returns:
            Dictionary with indexing statistics
        """
        if verbose:
            print(f"📂 Ingesting documents from: {directory_path}")
        
        # Ingest documents
        documents = DocumentIngester.ingest_directory(directory_path)
        if not documents:
            print("⚠️  No documents found")
            return {"error": "No documents found"}
        
        if verbose:
            print(f"✅ Ingested {len(documents)} documents/pages")
        
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(
                text=doc.content,
                source_doc=doc.file_name,
                page_number=doc.page_number
            )
            all_chunks.extend(chunks)
            if verbose:
                print(f"  {doc.file_name} (page {doc.page_number}): {len(chunks)} chunks")
        
        if verbose:
            print(f"✅ Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        if verbose:
            print("🔄 Generating embeddings...")
        
        texts_to_embed = [chunk.text for chunk in all_chunks]
        embeddings = self.embeddings.embed_batch(texts_to_embed)
        
        if verbose:
            print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Store in vector database
        self.vector_store.add_chunks(all_chunks, embeddings)
        
        stats = self.vector_store.get_collection_info()
        stats.update({
            "documents_ingested": len(documents),
            "chunks_created": len(all_chunks),
        })
        
        return stats
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query: Natural language query
            top_k: Number of results (uses instance default if None)
        
        Returns:
            List of retrieval results with scores and text
        """
        if top_k is None:
            top_k = self.top_k
        
        # Embed the query
        query_embedding = self.embeddings.embed(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, k=top_k)
        
        # Format results
        formatted_results = []
        for chunk_id, similarity, text, metadata in results:
            formatted_results.append({
                "chunk_id": chunk_id,
                "similarity_score": round(similarity, 4),
                "source_document": metadata.get("source_doc", "unknown"),
                "page_number": int(metadata.get("page_number", 0)),
                "text": text,
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return self.vector_store.get_collection_info()
    
    def clear_index(self) -> None:
        """Clear all indexed data."""
        self.vector_store.clear_collection()
        print("🗑️  Index cleared")

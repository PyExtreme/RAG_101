# Introduction to Vector Databases

Vector databases are specialized data storage systems designed to efficiently store and retrieve high-dimensional vectors. They have become essential infrastructure for modern machine learning and AI applications.

## What is a Vector Database?

A vector database stores vectors (sequences of numbers) and enables fast similarity search. Unlike traditional databases that retrieve data by exact keys or ranges, vector databases find data by similarity.

### Traditional Database vs Vector Database

**Traditional Database:**
- Optimized for: Exact matches, range queries, SQL operations
- Storage: Tables with rows and columns
- Queries: WHERE clauses on specific fields
- Speed: Linear scan for "similar" queries

**Vector Database:**
- Optimized for: Similarity search, nearest neighbors
- Storage: Vectors with metadata
- Queries: "Find vectors similar to this vector"
- Speed: Logarithmic or sub-linear via indexing

## Why Do We Need Vector Databases?

With the rise of embeddings and neural networks, traditional databases are inadequate:

1. **Scalability:** With millions of vectors, comparing all pairs is prohibitively slow
2. **High dimensionality:** Vectors often have hundreds or thousands of dimensions
3. **Similarity semantics:** Traditional databases don't understand semantic similarity
4. **Performance requirements:** Modern applications need sub-100ms query response times

## How Vector Databases Work

### Indexing Strategies

Vector databases use approximate nearest neighbor (ANN) algorithms to make search fast:

#### HNSW (Hierarchical Navigable Small World)
- Multi-layer graph structure
- Fast approximate nearest neighbor search
- Used by ChromaDB, Weaviate

#### IVF (Inverted File)
- Partitioning + local search
- Good balance of speed and memory
- Used by Faiss, Pinecone

#### LSH (Locality Sensitive Hashing)
- Hash vectors to buckets
- Vectors in same bucket likely similar
- Good for very large scale

#### Tree-based (ANNOY, KD-trees)
- Hierarchical space partitioning
- Simple, good for moderate scales

### Search Process

```
1. User query vector
      ↓
2. Load index into memory
      ↓
3. Use ANN algorithm to navigate index
      ↓
4. Examine candidate vectors
      ↓
5. Return top-k most similar
      ↓
6. Calculate exact distances if needed
```

### Trade-offs

Vector databases trade exactness for speed:
- **Exact search:** Compare with every vector (slow, 100% accurate)
- **Approximate search:** Use indexes (fast, ~99% accurate)

For most applications, approximate is good enough and much faster.

## ChromaDB Explained

ChromaDB is a lightweight vector database designed for:
- Ease of use (single library, no separate server)
- Python-first
- Persistent storage
- Built-in metadata support

### ChromaDB Architecture

```
┌─────────────────────────────┐
│   Application Code          │
│   (Python)                  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  ChromaDB Python API        │
│  - Collections              │
│  - Add, Query, Delete       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   HNSW Index                │
│   (Approximate Nearest      │
│    Neighbor Search)         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   DuckDB + Parquet          │
│   (Persistent Storage)      │
└─────────────────────────────┘
```

### Key Features

1. **Collections:** Organize vectors by topic/purpose
2. **Metadata:** Store and filter by document metadata
3. **HNSW indexing:** Fast approximate nearest neighbor search
4. **Persistence:** Save to disk, load on startup
5. **Metadata filtering:** Retrieve based on metadata and similarity

## Vector Database Landscape

### Options Available

| Database | Use Case | Cost | Scalability |
|----------|----------|------|-------------|
| ChromaDB | Learning, small projects | Free | Single machine |
| Pinecone | Production, cloud | $0.40/million vectors | Serverless |
| Weaviate | Flexible, hybrid search | Free/Paid | Self-hosted/Cloud |
| Milvus | Large scale | Free (self-hosted) | Distributed |
| Faiss | Research, very large | Free | CPU/GPU |
| Qdrant | Modern, Rust-based | Free/Paid | High performance |
| OpenSearch | Familiar (ES-like) | Free/Paid | Distributed |

### Choosing a Vector Database

Consider:
- **Scale:** How many vectors? (ChromaDB < 1M, Pinecone > 100M)
- **Cost:** Free vs. commercial
- **Features:** Metadata, filtering, multi-tenancy
- **Integration:** Does it fit your stack?
- **Team expertise:** Python-first vs. language-agnostic

## Common Use Cases

### 1. Semantic Search
```
Documents → Embeddings → Vector DB
Query → Embedding → Find most similar documents
```

### 2. Recommendation Systems
```
User preferences → Embeddings → Find similar items
```

### 3. Anomaly Detection
```
Normal data → Embeddings → Find outliers far from norm
```

### 4. Deduplication
```
All items → Embeddings → Find near-duplicates
```

### 5. Classification
```
Training data → Embeddings → Find nearest class
```

## Performance Considerations

### Indexing Time
- Depends on: Vector count, dimensionality, index type
- ChromaDB: ~1000 vectors/second on modern hardware

### Query Latency
- Depends on: Index type, vector count, exact/approximate
- ChromaDB: 1-50ms for typical queries

### Memory Usage
- Stored vectors: (count × dimensions × 4 bytes) for float32
- Indexes: Additional 10-30% overhead
- Example: 1M vectors × 768 dims = ~3GB vectors + 1GB index

### Optimization Techniques
1. **Batch inserts:** Faster than individual inserts
2. **Approximate search:** Trade accuracy for speed
3. **Metadata filtering:** Reduce search space
4. **Dimensionality reduction:** Reduce vector size
5. **Quantization:** Use lower precision (int8 instead of float32)

## Metadata and Filtering

Vector databases support metadata (text, numbers, timestamps):

```python
# Store embedding + metadata
vector_db.add(
    embeddings=[vector1, vector2],
    documents=["text1", "text2"],
    metadatas=[
        {"source": "doc1.pdf", "page": 1},
        {"source": "doc2.txt", "page": None}
    ]
)

# Query with metadata filtering
results = vector_db.query(
    query_embedding=query_vec,
    where={"source": "doc1.pdf"},  # Only search doc1
    n_results=5
)
```

## Hybrid Search

Modern vector databases support hybrid search:
- Combine semantic search with keyword filtering
- Example: "Find articles about embeddings from 2023"
- Semantic: Match meaning
- Keyword: Match metadata

## Challenges and Limitations

### Curse of Dimensionality
- High-dimensional spaces have unintuitive properties
- Distance becomes less meaningful
- Approximate algorithms become less effective

### Cold Start Problem
- New embeddings have no neighbors initially
- May take time for good recommendations

### Semantic Drift
- Meaning of documents evolves
- Old embeddings may become stale
- Need periodic re-embedding

### Storage Costs at Scale
- Billions of vectors require significant storage
- Compression and quantization help

## Future Trends

1. **Hybrid search:** Combining multiple modalities
2. **Real-time updates:** Faster re-indexing
3. **Distributed systems:** Scaling beyond single machine
4. **Cost optimization:** Compression, quantization
5. **Edge computing:** Vector search on devices

## Conclusion

Vector databases are becoming essential infrastructure for AI applications. They enable fast semantic search, recommendations, and other similarity-based operations at scale. Understanding how they work and their trade-offs is crucial for building modern AI systems.

# Week 2: Semantic Search from Scratch — Educational Guide

## Part 1: Core Concepts Explained Practically

### 1. Embeddings

**What are embeddings?**
Embeddings are vectors (lists of numbers) that represent the *semantic meaning* of text. Instead of treating words as discrete symbols, embeddings capture semantic relationships mathematically.

```
Text: "The cat sat on the mat"
          ↓
Embedding: [-0.15, 0.42, -0.08, ..., 0.71]  (768 numbers for nomic-embed-text)
```

**Why they work:**
- Words with similar meanings have similar embeddings
- Mathematical operations on embeddings reveal semantic relationships
- "cat" and "dog" embeddings are close to each other
- "cat" and "car" embeddings are farther apart

**Why enable similarity search:**
- Once text is a vector, you can measure distance between any two vectors
- Similar text = vectors close together in space
- This is the foundation of retrieval

**Practical implications in this project:**
```python
# Document chunk becomes embedding
chunk_text = "Machine learning is a subset of artificial intelligence"
embedding = [0.12, -0.45, 0.78, ...]  # 768 dimensions

# Query also becomes embedding
query = "What is machine learning?"
query_embedding = [0.11, -0.43, 0.76, ...]  # Same space!

# Now we can compare similarity
similarity = cosine(embedding, query_embedding) = 0.92  # Very similar!
```

**OpenAI vs Open-Source (Ollama) Trade-offs:**

| Aspect | OpenAI | Ollama (nomic-embed-text) |
|--------|--------|--------------------------|
| **Cost** | $0.02 per 1M tokens | FREE |
| **Privacy** | Data sent to OpenAI | Stays on your machine |
| **Quality** | Excellent (text-embedding-3-large) | Very good (768-dim) |
| **Speed** | Depends on network | Instant (local) |
| **Setup** | API key required | Download model once |

**For this project:** We use Ollama + nomic-embed-text for:
- ✅ Learning without costs
- ✅ Understanding the system end-to-end
- ✅ No API dependency
- ✅ Privacy (documents stay local)

---

### 2. Similarity Search

**What is it?**
Similarity search finds documents "most like" a query by measuring distance between vectors in embedding space.

**The three main metrics:**

#### Cosine Similarity (What we use)
```
cos(θ) = (A·B) / (||A|| * ||B||)
Range: -1 to 1 (typically 0 to 1 for embeddings)
```

**Why cosine?**
- Measures *angle* between vectors, not magnitude
- Two vectors pointing in same direction = high similarity, regardless of length
- Perfect for embeddings which can have different magnitudes
- Most popular in NLP/semantic search

**Interpretation:**
```
1.0  → Identical direction (same meaning)
0.8  → Very similar
0.5  → Moderately similar
0.2  → Somewhat different
0.0  → Orthogonal (unrelated)
```

#### Dot Product (Alternative)
```
A·B = Σ(a_i * b_i)
```

**When to use:**
- When vectors are normalized (Euclidean norm = 1)
- When you want raw magnitude + direction
- Faster computation (no normalization)

**Trade-off:**
- Magnitude-dependent (larger vectors score higher)
- Less intuitive

#### L2 (Euclidean Distance)
```
d = √(Σ(a_i - b_i)²)
```

**When to use:**
- When you want geometric distance
- When magnitudes matter

**Trade-off:**
- Slower in high dimensions
- Magnitudes affect score

**How it works in practice (in our engine):**

```python
# User query
query = "How do embeddings work?"

# 1. Convert to embedding
query_vec = embed(query) = [0.1, -0.5, 0.8, ..., -0.3]

# 2. Compare to all indexed chunks
chunk1_vec = embed("Embeddings are vectors representing text") = [0.09, -0.49, 0.79, ...]
chunk2_vec = embed("The weather is sunny today") = [-0.8, 0.1, 0.05, ...]

# 3. Calculate cosine similarity
similarity(query_vec, chunk1_vec) = 0.98  ✓ Very similar
similarity(query_vec, chunk2_vec) = 0.12  ✗ Not similar

# 4. Return chunks with highest similarity
```

---

### 3. Chunking Strategies

**Why chunking is required:**

Documents are usually too large for optimal embedding:
- Embeddings lose information on very long texts
- Vector databases work best with focused context
- Search becomes too broad without chunking
- Storage efficiency matters

**Problem:** If we embed entire document, we lose granularity. If someone asks "What's on page 5?", we can't pinpoint it.

**Solution:** Break into chunks, embed each chunk, track metadata.

#### Fixed-Size Chunking (What we implement)

```python
chunk_size = 500  # characters
overlap = 100     # characters

# Document:
# "Machine learning is... [400 chars] ... neural networks [400 chars] ... deep learning [400 chars]"
#
# Chunks:
# [0:500]   "Machine learning is... [500 chars]"
# [400:900] "[100 overlap] ... [400 new] ..."  
# [800:1300] ...
```

**Advantages:**
- ✅ Simple and predictable
- ✅ Consistent embedding quality
- ✅ Easy to implement

**Trade-offs:**
- ❌ May cut sentences in middle
- ❌ Loss of context at boundaries
- ❌ Not semantically aware

**Overlap impact:**
```
No overlap (100% -> 0%):
- Fewer chunks
- Less storage
- Lower recall (might miss relevant info at boundaries)

With overlap (100% -> 80%):
- Redundant chunks
- More storage
- Higher recall (captures context crossing boundaries)
```

#### Semantic Chunking (Alternative)

```python
# Chunks on sentence/paragraph boundaries, not character count
chunk = "Machine learning is a subset of AI. It enables systems to learn. Neural networks are popular."
# Split at sentence boundaries, not arbitrary position
```

**Advantages:**
- ✅ Preserves meaning
- ✅ Better retrieval quality
- ✅ Natural information units

**Trade-offs:**
- ❌ Variable chunk sizes
- ❌ May exceed embedding limits
- ❌ Requires sentence detection library

**In our project:** We use fixed-size (configurable) because:
- ✅ Predictable behavior
- ✅ Easier to learn
- ✅ Works well for most documents
- ✅ Can add overlap to improve recall

---

### 4. Vector Databases (ChromaDB)

**Why not traditional databases?**

Traditional databases (PostgreSQL, MongoDB):
- ❌ Not optimized for high-dimensional vectors
- ❌ Can't do efficient similarity search
- ❌ Sequential scan required (slow)

Vector databases (ChromaDB, Pinecone, Weaviate):
- ✅ Optimized for similarity search
- ✅ Approximate nearest neighbor (ANN) indexes
- ✅ Fast even with millions of vectors

**What ChromaDB does (under the hood):**

1. **Storage:** Stores embeddings + metadata efficiently
2. **Indexing:** Builds approximate nearest neighbor index (HNSW)
   - Not exact nearest neighbor (faster)
   - Very accurate (>99% for search)
3. **Querying:** Returns top-K most similar embeddings
4. **Persistence:** Saves to disk for later use

```python
# What happens when you search:
query_embedding = [0.1, -0.5, 0.8, ...]

# ChromaDB's HNSW index finds nearest neighbors quickly
# (not by comparing all vectors, but using smart indexing)

results = [
    (chunk_id_1, similarity=0.95, text="..."),
    (chunk_id_2, similarity=0.87, text="..."),
    (chunk_id_3, similarity=0.82, text="..."),
]
```

**Architecture in our project:**

```
Documents → Chunking → Embeddings → ChromaDB
                                        ↓
                                   Persistent Store
                                   (chroma_db/)

Query → Embed → Search ChromaDB → Results
```

**Why this design:**
- Separation of concerns (ingestion, embedding, storage)
- Easy to swap components
- Modular and testable

---

## Part 2: Project Design

### Data Flow

```
┌─────────────────┐
│   Documents     │
│ (PDF/TXT/MD)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion     │  Extract text from files
│  (src/ingestion)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  Split into pieces
│  (src/chunking) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  Convert to vectors
│  (src/embeddings)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  Store & index
│(src/vector_store)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Disk Storage   │
│ (chroma_db/)    │
└─────────────────┘

USER QUERY:
    │
    ▼
┌─────────────────┐
│   Embed Query   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search Index   │  Find similar chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Results      │  Return with metadata
└─────────────────┘
```

### Configuration Trade-offs

**Chunk Size (default 500):**
- Smaller (200): More chunks, granular results, more storage
- Larger (1000): Fewer chunks, more context, less storage

**Chunk Overlap (default 100):**
- None: Less storage, might miss context at boundaries
- High (200): Better coverage, 2x storage

**Top-K Results (default 5):**
- Smaller (1-3): Fastest, most confident results
- Larger (20+): More context, may include false positives

---

## Part 3: Learning Outcomes

After building this system, you should understand:

1. **How semantic similarity works**
   - Text → embeddings (vectors)
   - Vectors → distances (cosine, L2, dot product)
   - Distances → "most similar" results

2. **Why chunking matters**
   - Embeddings work better on shorter text
   - Chunking enables granular retrieval
   - Overlap improves recall

3. **Vector databases aren't magic**
   - They store and index embeddings
   - Efficient search via approximate algorithms
   - Trade-off between speed and accuracy

4. **Strengths & limitations**

   **Strengths:**
   - ✅ Find semantically related content
   - ✅ Works across word choices ("AI" vs "artificial intelligence")
   - ✅ Fast retrieval at scale

   **Limitations:**
   - ❌ Semantic ambiguity (query could mean multiple things)
   - ❌ False positives (unrelated but similar vectors)
   - ❌ Struggles with very specific queries
   - ❌ No understanding of negation ("NOT machine learning" might still retrieve ML docs)

---

## Part 4: Suggested Follow-up Experiments

1. **Chunk Size Sensitivity**
   ```bash
   # Try different chunk sizes and measure quality
   # Do smaller chunks give more relevant results?
   # How does it affect search speed?
   ```

2. **Overlap Impact**
   ```bash
   # Index with 0% overlap vs 20% vs 40%
   # Does overlap actually improve recall?
   # What's the storage cost?
   ```

3. **Similarity Metrics**
   ```bash
   # Compare cosine vs dot product vs L2
   # Do results change significantly?
   # Which is fastest on your hardware?
   ```

4. **Different Embedding Models**
   ```bash
   # Try different Ollama models
   # nomic-embed-text (768-dim)
   # mxbai-embed-large (1024-dim)
   # Does quality improve?
   ```

5. **False Positives**
   ```bash
   # Create tricky queries with false positives
   # Example: "What is NOT machine learning?"
   # How does the system perform?
   ```

6. **Real-world Documents**
   ```bash
   # Test with PDFs vs plain text vs markdown
   # Do complex layouts cause problems?
   # How do you handle tables/figures?
   ```

---

## Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `src/ingestion.py` | Load documents (PDF, TXT, MD) |
| `src/chunking.py` | Split text into chunks |
| `src/embeddings.py` | Generate embeddings via Ollama |
| `src/vector_store.py` | Store in ChromaDB |
| `src/similarity.py` | Measure vector similarity |
| `src/search_engine.py` | Main orchestrator |
| `app.py` | Streamlit web interface |

### Configuration

Edit `.env` or `.env.example`:
```env
OLLAMA_MODEL=nomic-embed-text          # Embedding model
OLLAMA_BASE_URL=http://localhost:11434 # Ollama server
CHUNK_SIZE=500                         # Characters per chunk
CHUNK_OVERLAP=100                      # Overlap between chunks
TOP_K=5                                # Results to retrieve
```

### Common Commands

```bash
# Setup
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull model (in another terminal)
ollama pull nomic-embed-text

# Run app
streamlit run app.py

# Index documents
# (Use Streamlit UI: Index Documents tab)

# Search
# (Use Streamlit UI: Search tab)
```

---

## Troubleshooting

**Error: "Cannot connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check URL in `.env` matches Ollama server
- Try: `curl http://localhost:11434/api/tags`

**Error: "Model not found"**
- Pull the model: `ollama pull nomic-embed-text`
- Check installed models: `ollama list`

**Search returns no results**
- Index documents first using UI
- Check that documents exist in `./data/documents`
- Verify ChromaDB isn't empty: check `./data/chroma_db`

**Slow searches**
- Reduce chunk size (faster but less context)
- Use fewer chunks per query (reduce K)
- Check Ollama isn't overloaded

---

## What's Next?

- **Week 3:** Add query expansion and multi-hop retrieval
- **Week 4:** Implement reranking for better results
- **Week 5:** Add LLM integration for answer generation

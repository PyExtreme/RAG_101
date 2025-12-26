# Semantic Search Engine from Scratch

**Week 2 Learning Track: Building Intuition and Hands-On Skills**

A complete, educational implementation of a semantic search system. Learn how embeddings, similarity metrics, chunking strategies, and vector databases work together to power modern retrieval systems.

## 🎯 Quick Start (< 10 minutes)

### 1. Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- ~2GB disk space for embeddings index

### 2. Setup

```bash
# Clone and navigate
cd /path/to/RAG_101

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy example config
cp .env.example .env
```

### 3. Start Ollama

In a separate terminal:

```bash
# Start Ollama server
ollama serve

# In another terminal, pull the embedding model
ollama pull nomic-embed-text
```

### 4. Run the App

```bash
streamlit run app.py
```

This opens a web app at `http://localhost:8501`

### 5. Try It Out

1. Click **"Index Documents"** tab
2. Click **"📂 Index Documents"** button (uses sample documents)
3. Click **"🔎 Search"** tab
4. Try queries like:
   - "What are embeddings?"
   - "How does machine learning work?"
   - "What is a vector database?"

**Done!** You've built a semantic search system.

---

## 📚 Understanding the System

### What You're Building

A complete **semantic search pipeline**:

```
Documents → Chunks → Embeddings → Vector DB → Search Results
```

1. **Documents:** PDF, TXT, MD files
2. **Chunks:** Break into semantic units
3. **Embeddings:** Convert to vectors via Ollama
4. **Vector DB:** Index in ChromaDB
5. **Search:** Find most similar chunks

### Key Concepts

#### 🧮 Embeddings
- Convert text → 768-dimensional vectors
- Similar text = similar vectors
- Enabled by Ollama + nomic-embed-text

**Why embeddings?**
- Enable semantic similarity calculations
- Capture meaning, not just keywords
- Much cheaper than API calls (free!)

#### 🔍 Similarity Search
- Measure distance between vectors
- Cosine similarity: angle between vectors
- Returns top-k most similar results

**Why cosine similarity?**
- Works regardless of vector magnitude
- Scale-invariant
- Industry standard for NLP

#### ✂️ Chunking
- Break documents into manageable pieces
- Trade-off: smaller chunks vs. context

**Chunk size impact:**
- Too small: loses context
- Too large: less relevant results
- Overlap: improves recall but increases storage

#### 🗄️ Vector Database
- Efficient storage of embeddings
- Fast similarity search via HNSW indexing
- Metadata support for filtering

**Why ChromaDB?**
- Free, lightweight, Python-native
- No separate server needed
- Perfect for learning and small-to-medium projects
- Easily swappable for Pinecone/Weaviate at scale

---

## 🏗️ Project Structure

```
RAG_101/
├── src/                          # Core modules
│   ├── __init__.py
│   ├── config.py                # Configuration
│   ├── ingestion.py             # Load PDF/TXT/MD
│   ├── chunking.py              # Split text strategically
│   ├── embeddings.py            # Generate embeddings via Ollama
│   ├── similarity.py            # Cosine, dot product, L2 metrics
│   ├── vector_store.py          # ChromaDB integration
│   └── search_engine.py         # Main orchestrator
├── data/
│   ├── documents/               # Put your PDFs/TXT/MD here
│   │   ├── machine_learning_intro.md
│   │   ├── embeddings_guide.md
│   │   └── vector_databases.md
│   └── chroma_db/               # ChromaDB storage (auto-created)
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── README.md                    # This file
└── LEARNING_GUIDE.md            # Detailed educational content
```

---

## 🔧 Configuration

Edit `.env` to customize behavior:

```env
# Ollama Configuration
OLLAMA_MODEL=nomic-embed-text                    # Embedding model
OLLAMA_BASE_URL=http://localhost:11434           # Ollama server URL

# ChromaDB Configuration
CHROMA_DB_PATH=./data/chroma_db                  # Where to store embeddings

# Search Configuration
TOP_K=5                                          # Number of results
CHUNK_SIZE=500                                   # Characters per chunk
CHUNK_OVERLAP=100                                # Overlap between chunks
```

### Configuration Trade-offs

| Setting | Impact |
|---------|--------|
| **CHUNK_SIZE** | Larger = more context but fewer results; Smaller = more granular but less context |
| **CHUNK_OVERLAP** | Larger = better recall but more storage; 0 = minimal storage |
| **TOP_K** | Larger = more results but slower; Smaller = faster but might miss relevant docs |
| **OLLAMA_MODEL** | Different models trade quality vs. speed |

---

## 📖 Usage Examples

### Via Web Interface (Streamlit)

```bash
streamlit run app.py
```

Three tabs:
1. **📤 Index Documents:** Upload documents, configure chunking
2. **🔎 Search:** Natural language queries
3. **📊 Stats:** View index statistics

### Via Python Code

```python
from src.search_engine import SemanticSearchEngine
from src.config import Config

# Initialize engine
engine = SemanticSearchEngine(
    persist_dir=Config.CHROMA_DB_PATH,
    embedding_provider="ollama",
    chunking_strategy="fixed",
    chunk_size=500,
    chunk_overlap=100,
    top_k=5
)

# Index documents
stats = engine.index_documents("./data/documents")
print(f"Indexed {stats['chunks_created']} chunks")

# Search
results = engine.search("What are embeddings?")

# Display results
for result in results:
    print(f"Score: {result['similarity_score']}")
    print(f"Source: {result['source_document']}")
    print(f"Text: {result['text'][:100]}...")
    print("---")
```

### Command Line (if you add a CLI)

```bash
python -m src.cli index ./data/documents
python -m src.cli search "Your query here"
```

---

## 🧪 Testing Different Configurations

### Experiment 1: Chunk Size Impact

```bash
# Test different chunk sizes
for size in 200 500 1000; do
    sed -i "s/CHUNK_SIZE=.*/CHUNK_SIZE=$size/" .env
    streamlit run app.py  # Re-run and observe
done
```

**Expected findings:**
- Smaller chunks: More results, more specific
- Larger chunks: Fewer results, more context
- Find your sweet spot (usually 300-700)

### Experiment 2: Overlap Impact

```bash
# Test impact of overlap
python -c "
from src.search_engine import SemanticSearchEngine

for overlap in [0, 50, 150]:
    engine = SemanticSearchEngine(chunk_overlap=overlap)
    engine.index_documents('./data/documents')
    results = engine.search('machine learning')
    print(f'Overlap {overlap}: {len(results)} results')
"
```

**Expected findings:**
- No overlap: Fewer chunks, potential gaps
- High overlap: More chunks, better coverage
- Storage trade-off matters at scale

### Experiment 3: Similarity Metrics

```python
from src.search_engine import SemanticSearchEngine

# Compare metrics
for metric in ["cosine", "dot_product", "euclidean"]:
    engine = SemanticSearchEngine(similarity_metric=metric)
    results = engine.search("embeddings")
    # Compare result rankings
```

---

## 📊 How It Works: Data Flow

### Indexing Flow

```
┌──────────────────────────────────────────────────────────┐
│ 1. INGESTION (ingestion.py)                              │
│    Load PDF/TXT/MD → Raw text                            │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 2. CHUNKING (chunking.py)                                │
│    Split text → Fixed-size pieces with overlap           │
│    E.g., 500 chars/chunk, 100 char overlap               │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 3. EMBEDDINGS (embeddings.py)                            │
│    Each chunk → 768-dim vector                           │
│    Via Ollama + nomic-embed-text                         │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 4. VECTOR STORE (vector_store.py)                        │
│    Store embeddings + metadata in ChromaDB               │
│    Build HNSW indexes for fast search                    │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. PERSISTENCE                                           │
│    Save to ./data/chroma_db                              │
│    Load on startup (no re-indexing needed)               │
└──────────────────────────────────────────────────────────┘
```

### Search Flow

```
┌──────────────────────────────────────────────────────────┐
│ 1. USER QUERY                                            │
│    "What is semantic search?"                            │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 2. EMBED QUERY (embeddings.py)                           │
│    Query → 768-dim vector (same space as documents)      │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 3. SEARCH (vector_store.py)                              │
│    Find top-K vectors closest to query                   │
│    Use HNSW index (fast approximate search)              │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 4. SIMILARITY (similarity.py)                            │
│    Calculate cosine similarity scores                    │
│    Range: 0.0 (unrelated) to 1.0 (identical)            │
└────────────────┬─────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. RETURN RESULTS                                        │
│    [                                                     │
│      {score: 0.87, source: "embeddings_guide.md", ...},│
│      {score: 0.81, source: "ml_intro.md", ...},         │
│      ...                                                │
│    ]                                                     │
└──────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Learnings

### 1. How Semantic Similarity Works

- Text → vectors (embeddings)
- Similar meaning → vectors close together
- Distance between vectors = dissimilarity score
- Cosine similarity = industry standard

### 2. Why Chunking Matters

- Embeddings work better on focused text
- Smaller chunks → more retrieval results
- Overlap → improves recall
- Trade-off between granularity and context

### 3. Vector Databases Are Essential

- Can't do similarity search efficiently with SQL
- Need specialized indexes (HNSW, IVF, etc.)
- ChromaDB provides this with minimal setup
- Scales from thousands to millions of vectors

### 4. Strengths of Semantic Search

✅ Finds semantically related content  
✅ Works across different phrasings  
✅ Fast at scale (with proper indexing)  
✅ No need for manual tagging/labels

### 5. Limitations to Know

❌ Semantic ambiguity (query could mean multiple things)  
❌ False positives (unrelated but similar vectors)  
❌ Struggles with negation ("NOT machine learning" still retrieves ML)  
❌ Requires good embeddings model

---

## 🚀 Extension Ideas

### 1. Add More Embedding Models

```python
# In src/embeddings.py, add support for:
- mxbai-embed-large (larger, better quality)
- all-MiniLM-L6-v2 (smaller, faster)
- Custom fine-tuned models
```

### 2. Implement Reranking

```python
# After initial search, rerank results with:
- Different similarity metric
- Cross-encoder model
- Custom scoring function
```

### 3. Add LLM Answer Generation

```python
# Use retrieved chunks to generate answers:
- Integrate with local LLM (Ollama)
- Create RAG (Retrieval Augmented Generation) pipeline
- Combine search + generation for QA
```

### 4. Web UI Improvements

```python
# Enhance Streamlit app:
- File upload for documents
- Show chunk relationships
- Visualize embedding space (UMAP/t-SNE)
- Export results as PDF
```

### 5. Scale to Production

```python
# Move to production:
- Switch to Pinecone for larger scale
- Add API endpoints (FastAPI)
- Implement caching
- Add authentication
- Set up continuous indexing
```

---

## 🔧 Troubleshooting

### "Cannot connect to Ollama"

**Symptom:** `RuntimeError: Cannot connect to Ollama at http://localhost:11434`

**Solutions:**
```bash
# 1. Check Ollama is running
ollama serve

# 2. Verify connectivity
curl http://localhost:11434/api/tags

# 3. Check .env has correct URL
grep OLLAMA_BASE_URL .env

# 4. On macOS, try different host
# Edit .env: OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### "Model not found"

**Symptom:** `Error: Model 'nomic-embed-text' not found`

**Solutions:**
```bash
# Pull the model
ollama pull nomic-embed-text

# Check installed models
ollama list

# Try different model
# Edit .env: OLLAMA_MODEL=all-MiniLM-L6-v2
# Then: ollama pull all-MiniLM-L6-v2
```

### "No documents found"

**Symptom:** Index succeeds but finds 0 documents

**Solutions:**
```bash
# 1. Check directory exists
ls ./data/documents

# 2. Add sample documents
# Already included in this repo

# 3. Check file extensions
# Must be .pdf, .txt, .md (case-sensitive on Linux/Mac)
```

### "Search returns no results"

**Symptom:** Search runs but no results found

**Solutions:**
```bash
# 1. Check index isn't empty
# Use Stats tab in Streamlit

# 2. Try simpler query
# Complex queries might not match anything

# 3. Reduce similarity threshold
# Modify vector_store.py to return lower-scored results

# 4. Check ChromaDB folder
ls -la ./data/chroma_db
```

### "Search is very slow"

**Symptom:** Queries take >5 seconds

**Solutions:**
```bash
# 1. Reduce chunk count
# Edit .env: TOP_K=3 (instead of 5)

# 2. Smaller chunks (faster indexing)
# Edit .env: CHUNK_SIZE=300 (instead of 500)

# 3. Reduce overlap
# Edit .env: CHUNK_OVERLAP=0 (instead of 100)

# 4. Check Ollama isn't overloaded
# Look at Ollama server console
```

---

## 📈 Performance Metrics

On typical modern hardware:

| Operation | Typical Time | Notes |
|-----------|------|-------|
| Embed 1000 chunks | 30-60 seconds | Depends on Ollama hardware |
| Index in ChromaDB | 5-10 seconds | 1000 chunks |
| Single query | 100-500ms | Depends on index size |
| Full pipeline (100 doc) | 2-5 minutes | Single thread |

---

## 📝 Example Queries to Try

Use the sample documents included (`machine_learning_intro.md`, `embeddings_guide.md`, `vector_databases.md`):

### Semantic Match Queries
```
"What are embeddings?"
"How do neural networks work?"
"Tell me about vector databases"
```

### Cross-document Queries
```
"What is the relationship between embeddings and machine learning?"
"How do vector databases enable semantic search?"
```

### Challenging Queries
```
"Benefits of semantic search"
"Machine learning vs deep learning"
"Embeddings in production"
```

### Likely False Positives
```
"Not machine learning"  # Will still find ML content
"Opposite of clustering"  # Might confuse
```

---

## 🏆 Design Decisions & Trade-offs

### 1. **Fixed-Size Chunking vs Semantic Chunking**
- **Chose:** Fixed-size
- **Why:** Simpler, more predictable, good learning tool
- **Alternative:** Semantic would preserve meaning better but requires sentence detection

### 2. **Ollama vs OpenAI Embeddings**
- **Chose:** Ollama (local)
- **Why:** Free, private, no API dependency, perfect for learning
- **Alternative:** OpenAI better quality but costs money

### 3. **ChromaDB vs Pinecone/Weaviate**
- **Chose:** ChromaDB
- **Why:** No separate server, lightweight, good for learning
- **Alternative:** Pinecone scales better but less educational

### 4. **Streamlit vs FastAPI/Flask**
- **Chose:** Streamlit
- **Why:** Rapid development, great for demos, easy to understand
- **Alternative:** FastAPI better for production APIs

### 5. **Cosine vs Dot Product Similarity**
- **Chose:** Cosine
- **Why:** Scale-invariant, industry standard, more intuitive
- **Alternative:** Dot product faster if vectors normalized

---

## 📚 Further Reading

### Core Concepts
- [What are Embeddings?](https://github.com/huggingface/course/blob/main/chapters/en/chapter1/2.mdx)
- [Understanding Vector Databases](https://www.datacamp.com/blog/vector-databases)
- [Semantic Search Explained](https://www.sbert.net/docs/usage/semantic_search.html)

### Implementation Details
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [HNSW Algorithm](https://arxiv.org/abs/1802.02413)

### Advanced Topics
- [Retrieval Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Embeddings Bias and Fairness](https://arxiv.org/abs/1907.09969)

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Modify configurations and experiment
- Add new chunking strategies
- Implement different similarity metrics
- Add new embedding providers
- Improve the Streamlit UI

---

## 📄 License

MIT License - feel free to use for learning and projects

---

## 🎓 What's Next?

**Week 3:** Multi-hop retrieval and query expansion  
**Week 4:** Reranking and relevance optimization  
**Week 5:** RAG (Retrieval Augmented Generation) with LLMs  
**Week 6:** Production deployment and scaling

---

## ❓ Questions?

Refer to:
- [LEARNING_GUIDE.md](LEARNING_GUIDE.md) - Detailed educational content
- Code comments - Each module has detailed docstrings
- This README - Overview and troubleshooting

Happy learning! 🚀

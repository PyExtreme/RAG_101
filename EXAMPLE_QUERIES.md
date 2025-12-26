# Example Queries and Expected Results

This document provides example queries you can try with the sample documents included in this project.

## Sample Documents Included

1. **machine_learning_intro.md** - Introduction to machine learning, neural networks, applications
2. **embeddings_guide.md** - Understanding embeddings, semantic similarity, embedding models
3. **vector_databases.md** - Vector databases, ChromaDB, indexing strategies

## Example Queries

### ✅ Good Matches (High Similarity Expected)

These queries will find highly relevant results because they match content directly:

```
Query: "What are embeddings?"
Expected: embeddings_guide.md - Direct explanation of embeddings
Similarity: 0.85-0.95

Query: "How does machine learning work?"
Expected: machine_learning_intro.md - Core ML concepts
Similarity: 0.80-0.92

Query: "What is ChromaDB?"
Expected: vector_databases.md - Specific section on ChromaDB
Similarity: 0.82-0.88

Query: "Neural networks"
Expected: machine_learning_intro.md - Section on neural networks
Similarity: 0.78-0.90

Query: "Semantic similarity"
Expected: embeddings_guide.md - Detailed explanation
Similarity: 0.85-0.95
```

### 🟡 Moderate Matches (Medium Similarity)

These queries will find related but not exact matches:

```
Query: "How do vectors help with text?"
Expected: Mix of embeddings_guide.md and vector_databases.md
Similarity: 0.65-0.80
Why: Related but doesn't use exact same words

Query: "Deep learning applications"
Expected: machine_learning_intro.md (has both concepts)
Similarity: 0.70-0.82
Why: Both terms present but not always together

Query: "Storage and indexing"
Expected: vector_databases.md
Similarity: 0.68-0.78
Why: Related to indexing but different focus

Query: "Comparing different models"
Expected: embeddings_guide.md (compares Word2Vec, BERT, etc.)
Similarity: 0.65-0.75
Why: General concept, not specific phrase
```

### 🔴 Challenging Queries (Lower Similarity)

These queries are harder because they require more inference:

```
Query: "What's the opposite of clustering?"
Expected: May find clustering content, but not its opposite
Similarity: 0.45-0.65
Challenge: Negation is hard for embeddings

Query: "NOT machine learning"
Expected: Will still find ML content (can't handle NOT)
Similarity: 0.50-0.70
Challenge: Negative queries confuse embedding-based search

Query: "Advantages and disadvantages of using APIs"
Expected: May find relevant content (embeddings mentioned API-based services)
Similarity: 0.55-0.75
Challenge: Specific domain concept, not directly covered

Query: "How fast is semantic search?"
Expected: Will find vector database content about speed
Similarity: 0.60-0.75
Challenge: Performance is mentioned but not deeply discussed
```

### 🎯 Cross-Document Queries

These queries require combining concepts from multiple documents:

```
Query: "How do embeddings enable semantic search?"
Expected: Content from both embeddings_guide.md and vector_databases.md
Similarity: 0.75-0.85
Why: Combines concepts across documents

Query: "Machine learning with vector databases"
Expected: Both machine_learning_intro.md and vector_databases.md
Similarity: 0.70-0.82
Why: Bridges two documents

Query: "Comparing different similarity metrics for search"
Expected: similarity.py comments + vector_databases.md
Similarity: 0.68-0.80
Why: Requires understanding from multiple sources
```

## Understanding the Results

### Similarity Score Interpretation

```
0.90-1.00  → Excellent match (direct answer)
0.80-0.89  → Very good match (highly relevant)
0.70-0.79  → Good match (relevant)
0.60-0.69  → Fair match (somewhat related)
0.50-0.59  → Weak match (tangentially related)
< 0.50     → Poor match (likely irrelevant)
```

### Why Some Queries Return Lower Scores

1. **Semantic gap:** Query meaning differs from document content
2. **Vocabulary mismatch:** Different words used for same concept
3. **Abstraction level:** Very general vs specific queries
4. **Negation:** "NOT" and "except" are hard for embeddings
5. **Ambiguity:** Query could mean multiple things

## Experiments to Try

### 1. Same Meaning, Different Words

```
Query A: "What are embeddings?"
Query B: "Explain vector representations of text"
Query C: "How do you convert words to numbers?"

Expected: All should find similar results with similar scores
This demonstrates semantic understanding!
```

### 2. Chunk Size Impact

Try the same query with different chunk sizes:

```bash
# Edit .env to try different sizes
CHUNK_SIZE=300  # Smaller, more chunks
CHUNK_SIZE=500  # Default
CHUNK_SIZE=1000 # Larger, fewer chunks

# Reindex and search
# Do results change? How does score change?
```

### 3. Overlap Impact

```bash
# Try different overlap settings
CHUNK_OVERLAP=0    # No overlap
CHUNK_OVERLAP=50   # 10% overlap
CHUNK_OVERLAP=200  # 40% overlap

# Search for same query
# Does overlap affect which results are found?
# Does it improve coverage of concepts?
```

### 4. Similarity Metric Comparison

```
# In Streamlit UI, you'd modify to support this
# For manual testing:

from src.embeddings import EmbeddingFactory
from src.similarity import SimilarityMetricFactory

embedding = EmbeddingFactory.create()
query_vec = embedding.embed("What are embeddings?")

# Compare results with different metrics
for metric in ["cosine", "dot_product", "euclidean"]:
    similarity = SimilarityMetricFactory.create(metric)
    # Score different chunks
```

## Debugging Lower-Than-Expected Scores

If you get lower scores than expected:

1. **Check the document exists**
   - Verify the file is in `./data/documents/`
   - Check file extension (.md, .txt, .pdf)

2. **Verify indexing worked**
   - Check Stats tab shows chunks
   - Look at `./data/chroma_db/` folder size

3. **Try simpler queries**
   - Start with exact phrases from documents
   - Then try more abstract concepts

4. **Check chunk boundaries**
   - A concept might be split across chunks
   - Try searching for parts of the phrase

5. **Inspect embeddings directly**
   ```python
   from src.embeddings import EmbeddingFactory
   
   embedder = EmbeddingFactory.create()
   
   # Get embeddings for comparison
   vec1 = embedder.embed("machine learning")
   vec2 = embedder.embed("artificial intelligence")
   
   # Calculate similarity
   from src.similarity import SimilarityMetricFactory
   cosine = SimilarityMetricFactory.create("cosine")
   score = cosine.similarity(vec1, vec2)
   print(f"Similarity: {score}")
   # Should be high since ML is a type of AI
   ```

## Real-World Query Examples

Try these realistic queries:

```
"Explain how embeddings work"
"What are the benefits of deep learning?"
"How do vector databases improve search?"
"List advantages of supervised learning"
"When should I use semantic search instead of keyword search?"
"Tell me about BERT and transformers"
"What is HNSW indexing?"
```

## Comparing Results Across Settings

Create a comparison table:

```
Query: "What are embeddings?"

| Setting | Chunks | Avg Score | Top Result |
|---------|--------|-----------|------------|
| Default | 45 | 0.82 | embeddings_guide.md chunk 1 |
| Size 300 | 89 | 0.79 | embeddings_guide.md chunk 2 |
| Size 1000 | 23 | 0.85 | embeddings_guide.md chunk 1 |
```

This helps you understand the trade-offs!

## Next Steps

1. **Run the app:** `streamlit run app.py`
2. **Index documents:** Use "Index Documents" tab
3. **Try queries:** Use "Search" tab with examples above
4. **Experiment:** Change settings and observe results
5. **Learn:** Read LEARNING_GUIDE.md to understand why results look the way they do

Happy experimenting! 🚀

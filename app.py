"""Streamlit web application for semantic search engine."""
import streamlit as st
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.search_engine import SemanticSearchEngine
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5em; color: #1f77b4; }
    .result-card {
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .similarity-badge {
        display: inline-block;
        padding: 5px 10px;
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = []


def initialize_engine():
    """Initialize the search engine."""
    try:
        return SemanticSearchEngine(
            persist_dir=Config.CHROMA_DB_PATH,
            embedding_provider="ollama",
            chunking_strategy="fixed",
            similarity_metric="cosine",
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            top_k=Config.TOP_K,
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize engine: {e}")
        st.error("Make sure Ollama is running: `ollama serve`")
        st.error(f"And pull the model: `ollama pull {Config.OLLAMA_MODEL}`")
        return None


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("# 🔍 Semantic Search Engine", unsafe_allow_html=True)
    st.markdown("""
    **Week 2: Building Semantic Search from Scratch**
    
    Upload documents, index them with embeddings, and search semantically!
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=1000,
            value=Config.CHUNK_SIZE,
            step=100,
            help="Larger chunks = more context, fewer chunks"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=chunk_size - 100,
            value=Config.CHUNK_OVERLAP,
            step=50,
            help="Overlap improves recall but increases storage"
        )
        
        top_k = st.slider(
            "Top Results (k)",
            min_value=1,
            max_value=20,
            value=Config.TOP_K,
            help="Number of most similar chunks to retrieve"
        )
        
        st.markdown("---")
        st.markdown("## 📚 About This Project")
        with st.expander("What is Semantic Search?"):
            st.markdown("""
            **Semantic Search** finds documents based on *meaning*, not keywords.
            
            **How it works:**
            1. Convert text → embeddings (vectors)
            2. Query also becomes a vector
            3. Find vectors closest to query (cosine similarity)
            4. Return most relevant chunks
            
            **Why embeddings?**
            - Capture semantic meaning in numbers
            - Enable similarity calculations
            - Work across different word choices
            """)
        
        with st.expander("Why Chunking?"):
            st.markdown("""
            **Chunking** breaks documents into small pieces.
            
            **Why?**
            - Embeddings work best on smaller text
            - Improves search relevance
            - Reduces storage overhead
            
            **Trade-offs:**
            - Smaller chunks: more relevant but more to store
            - Overlap: improves recall but increases storage
            """)
        
        with st.expander("Embeddings vs Similarity"):
            st.markdown("""
            **Embeddings:** Convert text to vectors
            - nomic-embed-text: 768-dimensional vectors
            - Captures semantic meaning
            
            **Similarity:** Measure how close vectors are
            - Cosine similarity: angle between vectors (0 to 1)
            - Scale-invariant, most popular for embeddings
            - 1.0 = identical meaning, 0.0 = unrelated
            """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Index Documents", "🔎 Search", "📊 Stats"])
    
    # Tab 1: Indexing
    with tab1:
        st.markdown("### Upload & Index Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            doc_path = st.text_input(
                "Document Directory Path",
                value="./data/documents",
                help="Path to folder containing PDF, TXT, or MD files"
            )
        
        with col2:
            if st.button("📂 Index Documents", use_container_width=True):
                st.session_state.engine = initialize_engine()
                
                if st.session_state.engine:
                    with st.spinner("🔄 Indexing documents..."):
                        try:
                            stats = st.session_state.engine.index_documents(
                                doc_path,
                                verbose=True
                            )
                            
                            if "error" not in stats:
                                st.session_state.indexed = True
                                st.success("✅ Documents indexed successfully!")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Documents", stats.get("documents_ingested", 0))
                                with col2:
                                    st.metric("Chunks Created", stats.get("chunks_created", 0))
                                with col3:
                                    st.metric("Total Stored", stats.get("total_chunks", 0))
                                with col4:
                                    st.metric("Storage Path", Path(stats.get("persist_directory", "")).name)
                            else:
                                st.error(stats.get("error", "Unknown error"))
                        
                        except Exception as e:
                            st.error(f"❌ Indexing failed: {e}")
    
    # Tab 2: Search
    with tab2:
        st.markdown("### Semantic Search")
        
        if not st.session_state.engine:
            st.session_state.engine = initialize_engine()
        
        if not st.session_state.indexed and st.session_state.engine:
            st.info("💡 Tip: Index documents first using the 'Index Documents' tab")
        
        # Search input
        query = st.text_area(
            "Enter your search query",
            placeholder="Example: What is machine learning?",
            height=100,
            help="Ask a natural language question or describe what you're looking for"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Search", use_container_width=True):
                if not query.strip():
                    st.warning("⚠️  Please enter a search query")
                elif not st.session_state.engine:
                    st.error("❌ Engine not initialized")
                else:
                    with st.spinner("🔄 Searching..."):
                        try:
                            st.session_state.search_results = st.session_state.engine.search(
                                query=query,
                                top_k=top_k
                            )
                            st.success(f"✅ Found {len(st.session_state.search_results)} results")
                        except Exception as e:
                            st.error(f"❌ Search failed: {e}")
        
        with col2:
            if st.button("🗑️  Clear Index", use_container_width=True):
                if st.session_state.engine:
                    st.session_state.engine.clear_index()
                    st.session_state.indexed = False
                    st.session_state.search_results = []
                    st.success("✅ Index cleared")
        
        # Display search results
        if st.session_state.search_results:
            st.markdown("### Search Results")
            
            for i, result in enumerate(st.session_state.search_results, 1):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        similarity = result["similarity_score"]
                        # Color code by similarity
                        if similarity >= 0.7:
                            color = "🟢"
                        elif similarity >= 0.5:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        st.markdown(f"{color} **#{i}**")
                        st.markdown(f"**Score:** {similarity}")
                    
                    with col2:
                        st.markdown(f"**Source:** {result['source_document']}")
                        if result['page_number'] > 0:
                            st.markdown(f"Page {result['page_number']}")
                        st.markdown("---")
                        st.write(result["text"])
                    
                    st.divider()
    
    # Tab 3: Statistics
    with tab3:
        st.markdown("### Index Statistics")
        
        if not st.session_state.engine:
            st.session_state.engine = initialize_engine()
        
        if st.session_state.engine:
            try:
                stats = st.session_state.engine.get_stats()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Collection Name", stats.get("collection_name", "N/A"))
                    st.metric("Total Chunks Indexed", stats.get("total_chunks", 0))
                
                with col2:
                    st.metric("Storage Location", Path(stats.get("persist_directory", "")).name)
                
                with st.expander("📋 Full Statistics"):
                    st.json(stats)
            
            except Exception as e:
                st.error(f"Could not load statistics: {e}")
        
        # Learning outcomes section
        st.markdown("---")
        st.markdown("### 🎓 What You're Learning")
        
        learning_points = [
            "**Embeddings:** Text → vectors that capture meaning",
            "**Chunking:** Breaking documents for optimal retrieval",
            "**Similarity:** Measuring vector closeness (cosine similarity)",
            "**Vector Databases:** Efficient storage and retrieval",
            "**Trade-offs:** Chunk size vs storage vs recall",
        ]
        
        for point in learning_points:
            st.markdown(f"- {point}")


if __name__ == "__main__":
    main()

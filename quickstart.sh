#!/bin/bash

# Semantic Search Engine - Quick Start Script

set -e

echo "🚀 Semantic Search Engine - Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if Ollama is running
echo "🔍 Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is not running"
    echo "   Start it with: ollama serve"
    echo "   Then run this script again"
    exit 1
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Open another terminal and run: ollama serve"
echo "2. In a third terminal, pull the model: ollama pull nomic-embed-text"
echo "3. Come back here and run: streamlit run app.py"
echo ""
echo "Then:"
echo "- Open http://localhost:8501 in your browser"
echo "- Go to 'Index Documents' tab"
echo "- Click 'Index Documents' button"
echo "- Go to 'Search' tab"
echo "- Try a query!"
echo ""

# 🚀 Getting Started Checklist

Complete this checklist to get your semantic search system running in under 10 minutes.

## ✅ Pre-Installation (1 minute)

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Ollama downloaded from https://ollama.ai
- [ ] You're in the RAG_101 directory
- [ ] Have ~2GB free disk space

## ✅ Step 1: Install Dependencies (2 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

Check:
- [ ] Virtual environment activated (you see `(venv)` in prompt)
- [ ] Command completes without errors
- [ ] Can import chromadb: `python3 -c "import chromadb"`

## ✅ Step 2: Start Ollama (1 minute)

**In a NEW terminal** (keep it running):

```bash
ollama serve
```

You should see:
```
Listening on 127.0.0.1:11434
```

Check:
- [ ] Ollama started successfully
- [ ] Terminal shows it's listening
- [ ] No error messages

## ✅ Step 3: Pull the Model (2-5 minutes)

**In ANOTHER NEW terminal**:

```bash
ollama pull nomic-embed-text
```

You should see:
```
pulling manifest
pulling f937e6386d9e
downloading...
✅ success
```

Check:
- [ ] Model download completes
- [ ] Says "success" at the end
- [ ] ~200MB downloaded (depending on connection)

## ✅ Step 4: Run the Application (1 minute)

**Back to original terminal** (with venv activated):

```bash
streamlit run app.py
```

You should see:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Check:
- [ ] Streamlit starts without errors
- [ ] Shows "Local URL: http://localhost:8501"
- [ ] Terminal stays running

## ✅ Step 5: Open in Browser (30 seconds)

- [ ] Open http://localhost:8501 in your browser
- [ ] You see the semantic search interface
- [ ] Three tabs visible: Index Documents, Search, Stats

## ✅ Step 6: Index Documents (1 minute)

1. Click **"Index Documents"** tab
2. Click **"📂 Index Documents"** button
3. Wait for completion (should say "✅ Documents indexed successfully!")

Check:
- [ ] See progress messages
- [ ] Successfully says "✅ Documents indexed"
- [ ] Shows numbers for Documents/Chunks/Total

## ✅ Step 7: Test Search (1 minute)

1. Click **"Search"** tab
2. In the search box, enter: `What are embeddings?`
3. Click **"🔍 Search"** button
4. View the results

You should see:
- Results with similarity scores (0.0-1.0)
- Source documents
- Relevant text excerpts

Check:
- [ ] Search completes without errors
- [ ] Shows results
- [ ] Similarity scores displayed
- [ ] Results make sense

## 🎉 Success!

If you've completed all steps:

✅ Your semantic search system is **running**  
✅ You've **indexed documents**  
✅ You've **performed a search**  

**Congratulations! You have a working semantic search engine!**

---

## 📚 Next Steps

### Quick (5 minutes)
- [ ] Read QUICK_REFERENCE.md
- [ ] Try more queries from EXAMPLE_QUERIES.md
- [ ] Check Stats tab to see index info

### Short (30 minutes)
- [ ] Read README.md (overview)
- [ ] Try different chunk sizes in config
- [ ] Index your own documents

### Medium (1-2 hours)
- [ ] Read LEARNING_GUIDE.md (deep dive)
- [ ] Open Jupyter notebook: `jupyter notebook Semantic_Search_Complete_Learning.ipynb`
- [ ] Run the experiments

### Complete (4-5 hours)
- [ ] Work through entire Jupyter notebook
- [ ] Understand each module in src/
- [ ] Try code examples from QUICK_REFERENCE.md
- [ ] Plan extensions from PROJECT_SUMMARY.md

---

## 🐛 Troubleshooting Quick Fixes

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running in another terminal:
ollama serve

# Check it works:
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Pull the model:
ollama pull nomic-embed-text

# Verify it's installed:
ollama list
```

### "No documents found"
```bash
# Check documents exist:
ls ./data/documents/

# Should show 3 markdown files
```

### "Search returns no results"
- Make sure you indexed first (Index Documents tab)
- Try simpler queries
- Check Stats tab shows chunks were created

### Still stuck?
See **README.md** → Troubleshooting section

---

## 📞 Finding Information

| Question | Where to Look |
|----------|--------------|
| How do embeddings work? | LEARNING_GUIDE.md |
| How do I use the app? | README.md → Usage Examples |
| What queries can I try? | EXAMPLE_QUERIES.md |
| How do I modify code? | QUICK_REFERENCE.md |
| How does it all fit together? | PROJECT_SUMMARY.md |
| How do I learn hands-on? | Jupyter Notebook |

---

## ✨ Tips for Success

1. **Be Patient:** Embedding generation takes a few seconds
2. **Check Ollama:** Watch the Ollama terminal to see progress
3. **Read Examples:** EXAMPLE_QUERIES.md shows what works
4. **Experiment:** Try different configurations
5. **Ask Questions:** Check documentation before giving up

---

## 🎯 Verification Commands

Verify each step completed:

```bash
# 1. Python installed
python3 --version

# 2. Ollama running
curl http://localhost:11434/api/tags

# 3. Model available
ollama list | grep nomic-embed-text

# 4. Dependencies installed
python3 -c "import chromadb; import streamlit"

# 5. App can start
streamlit run app.py --client.showErrorDetails=true
```

---

## 📊 Expected Output

When everything works:

```
✅ Python version: 3.10.0 (or higher)
✅ Ollama response: {"models": [...]}
✅ Model listed: nomic-embed-text 768B
✅ Imports succeed: No errors
✅ App starts: "Local URL: http://localhost:8501"
```

---

## 🚀 Ready to Go!

You now have:
- ✅ Working semantic search system
- ✅ Web interface for testing
- ✅ Sample documents indexed
- ✅ Understanding of how it works
- ✅ Roadmap for learning more

**Next:** Open browser → Try more queries!

Good luck! 🎓✨

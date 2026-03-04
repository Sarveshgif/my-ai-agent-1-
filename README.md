# 🤖 Local PDF AI Agent

A privacy-first AI agent that lives on your MacBook. It uses a local LLM to chat with your PDFs using Hybrid Search (Semantic + Keywords) so you can get answers from your documents without any API costs.

---

## 🚀 Getting Started

### 1. Setup the Environment
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the AI (Ollama)
```bash
ollama pull qwen2.5:3b
```

### 3. Run the Project
```bash
python3 indexer.py  # Index your PDFs in /docs
python3 ui.py       # Start the Chat UI
```

---

## 🛑 Resource Cleanup
To instantly free up your MacBook's RAM:
```bash
ollama stop qwen2.5:3b
```

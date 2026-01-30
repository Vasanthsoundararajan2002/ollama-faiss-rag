# employee-rag-ollama

RAG (Retrieval-Augmented Generation) system for employee document search using FAISS and Ollama LLaMA 3B.

## Project Architecture
```
Documents → Embeddings → FAISS Index
                ↓
User Query → Vector Search → Retrieved Docs
                ↓
Retrieved Docs + Query → Ollama LLM → Answer
```

## Features
- ✅ FAISS vector database for fast similarity search
- ✅ Processes 5 employee PDF files
- ✅ Small chunk size (500 chars) for better retrieval accuracy
- ✅ Local Ollama LLaMA 3B model (no API keys needed)
- ✅ Pure Python implementation (no LangChain)

## Project Structure
```
employee-rag-ollama/
│
├── pdf_loader.py          # Load PDFs, create embeddings, build FAISS index
├── query.py               # Ask questions and get answers
├── requirements.txt       # Python dependencies
├── data/                  # Place your employee PDF files here
└── faiss_index/          # Generated FAISS index (auto-created)
```

## Setup

### 1. Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.ai
```

### 2. Pull LLaMA model
```bash
ollama pull llama3.2:3b
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare your data
```bash
mkdir data
# Place your 5 employee PDF files in the data/ folder
```

## Usage

### Step 1: Process PDFs and create FAISS index
```bash
python pdf_loader.py
```
This will:
- Read all PDFs from `data/` folder
- Split text into 500-character chunks
- Create embeddings using sentence-transformers
- Build and save FAISS index

### Step 2: Ask questions
```bash
# Make sure Ollama is running
ollama serve

# Run query interface
python query.py
```

### Example Questions:
- "What is John's designation?"
- "Tell me about employee experience"
- "Who works in the marketing department?"
- "What are the skills of employees?"

Type `exit` to quit.

## How It Works

1. **PDF Loading**: Extracts text from employee PDFs
2. **Chunking**: Splits text into small 500-character chunks for accuracy
3. **Embedding**: Converts chunks to vector embeddings
4. **Indexing**: Stores embeddings in FAISS for fast retrieval
5. **Query**: User asks a question
6. **Search**: Finds top 3 most relevant chunks using vector similarity
7. **Generation**: Sends retrieved context + query to Ollama LLaMA
8. **Answer**: Returns AI-generated response

## Technical Stack
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Ollama LLaMA 3.2 (3B parameters)
- **PDF Processing**: PyPDF2
- **Language**: Python 3.8+

## Why Small Chunks?
Small chunk sizes (500 chars) provide:
- Better retrieval accuracy
- More precise context matching
- Reduced noise in responses
- Faster processing

## Requirements
- Python 3.8+
- Ollama installed locally
- At least 4GB RAM
- 2GB disk space for model

## Troubleshooting

**Ollama connection error:**
```bash
# Start Ollama service
ollama serve
```

**Model not found:**
```bash
# Pull the model again
ollama pull llama3.2:3b
```

**No PDFs found:**
- Make sure PDF files are in the `data/` folder
- Check file extensions are `.pdf`

## License
MIT

## Author
[Your Name]

## Feedback
Found this useful? Star ⭐ the repo!

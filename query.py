import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
import json

# Load FAISS index and chunks
print("Loading FAISS index...")
index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def search_relevant_chunks(query, top_k=3):
    """Search for most relevant chunks"""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def ask_ollama(prompt):
    """Send query to Ollama LLaMA model"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()['response']

# Main query loop
print("\n=== RAG System Ready ===")
print("Ask questions about employees (type 'exit' to quit)\n")

while True:
    question = input("Your question: ")
    
    if question.lower() == 'exit':
        break
    
    # Get relevant chunks
    print("\nSearching knowledge base...")
    relevant_docs = search_relevant_chunks(question, top_k=3)
    
    # Create context
    context = "\n\n".join(relevant_docs)
    
    # Create prompt for Ollama
    prompt = f"""Based on the following information, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    # Get answer from Ollama
    print("Generating answer...\n")
    answer = ask_ollama(prompt)
    print(f"Answer: {answer}\n")
    print("-" * 50)
```

## 3. requirements.txt
```
PyPDF2
faiss-cpu
sentence-transformers
numpy
requests

import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load PDFs
pdf_folder = "data/"
all_chunks = []

for file in os.listdir(pdf_folder):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, file)
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split into small chunks (500 characters)
            chunk_size = 500
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                    all_chunks.append(chunk)

print(f"Total chunks created: {len(all_chunks)}")

# Create embeddings
print("Creating embeddings...")
embeddings = embedding_model.encode(all_chunks)
embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and chunks
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/index.faiss")
with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("FAISS index saved successfully!")

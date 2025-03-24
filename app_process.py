import os
from dotenv import load_dotenv
import pickle
from flask import Flask, request, render_template
from PyPDF2 import PdfReader  # Replaced PyMuPDF with PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import faiss
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
DATA_DIR = "data"
UPLOAD_DIR = "uploads"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure uploads directory exists
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def generate_embeddings(texts):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    response = requests.post(
        "https://api.mistral.ai/v1/embeddings",
        headers=headers,
        json={"input": texts, "model": "mistral-embed"}
    )
    return [item['embedding'] for item in response.json()['data']]

@app.route('/', methods=['GET', 'POST'])
def upload_document():
    if request.method == 'POST':
        file = request.files['file']
        
        # Save file with secure path
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(file_path)
        
        # Process text
        text = ""
        if file.filename.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in reader.pages)
        else:
            # Handle text files with proper encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Fallback for problematic encodings
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        # Generate embeddings and save
        embeddings = generate_embeddings(chunks)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype('float32'))
        
        # Save to data directory
        faiss.write_index(index, os.path.join(DATA_DIR, "faiss_index.index"))
        with open(os.path.join(DATA_DIR, "text_chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        
        return "Document processed successfully!"
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=5000, debug=False)
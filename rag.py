import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔹 Stap 1: Tekst extractie uit PDF's
def extract_text_from_pdf(pdf_path):
    """Extraheert tekst uit een PDF-bestand."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

def extract_text_from_directory(directory_path):
    """Doorloopt een map en extraheert tekst uit alle PDF-bestanden."""
    pdf_texts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            pdf_texts[filename] = extract_text_from_pdf(pdf_path)
    return pdf_texts

# 🔹 Stap 2: Chunking
def split_text_into_chunks(text, chunk_size=300, chunk_overlap=50, filename=""):
    """Splitst een tekst in kleinere stukken voor efficiënte verwerking."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return [{"chunk": chunk, "filename": filename} for chunk in chunks]

# 🔹 Stap 3: Verwerk de PDF's en maak chunks
directory_path = "verweerschriften"  # Pas dit aan naar jouw map
pdf_texts = extract_text_from_directory(directory_path)

all_chunks = []
metadata = []
for filename, text in pdf_texts.items():
    chunk_info = split_text_into_chunks(text, chunk_size=300, chunk_overlap=50, filename=filename)
    for chunk_data in chunk_info:
        all_chunks.append(chunk_data["chunk"])
        metadata.append({"filename": chunk_data["filename"]})

# 🔹 Stap 4: Embed de chunks en bouw FAISS database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Lokaal draaien
vector_db = FAISS.from_texts(all_chunks, embeddings, metadatas=metadata)

# 🔹 Stap 5: Similary Search Query
def search_query(query, k=3):
    """Zoekt de meest relevante chunks op basis van een query."""
    query_embedding = embeddings.embed_query(query)
    results = vector_db.similarity_search_by_vector(query_embedding, k=k)
    return results

# 🔹 Voorbeeldquery
query = "Wat is de definitie van een event?"
results = search_query(query)

# 🔹 Print resultaten
for i, res in enumerate(results):
    print(f"🔹 Resultaat {i+1}: (Bestand: {res.metadata['filename']})\n{res.page_content}\n")

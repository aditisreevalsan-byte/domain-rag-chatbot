import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# Load HuggingFace Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdfs(uploaded_files):
    """Extracts text and page metadata from uploaded PDFs and chunks them."""
    chunks = []
    metadata = [] # stores {'source': filename, 'page': page_num}

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Split page text into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=120
                )
                page_chunks = text_splitter.split_text(text)
                for chunk in page_chunks:
                    chunks.append(chunk)
                    metadata.append({
                        "source": uploaded_file.name,
                        "page": i + 1
                    })
    
    if not chunks:
        return None, None, None

    # Create embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    # Initialize FAISS vector store
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    return index, chunks, metadata

def retrieve_context(query, index, chunks, metadata, top_k=3):
    """Retrieves top-k relevant chunks based on query embedding similarity."""
    query_vector = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)

    retrieved_chunks = []
    retrieved_meta = []
    
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            retrieved_meta.append(metadata[idx])
            
    return retrieved_chunks, retrieved_meta

def generate_answer(query, retrieved_chunks, retrieved_meta, api_key):
    """Generates grounded answer using Google Gemini API."""
    if not api_key:
        return "Please enter a valid Gemini API Key in the sidebar.", []

    client = genai.Client(api_key=api_key)

    context_str = ""
    sources_str = []
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_meta)):
        context_str += f"\n--- Excerpt {i+1} (Source: {meta['source']}, Page: {meta['page']}) ---\n{chunk}\n"
        sources_str.append(f"{meta['source']} (Page {meta['page']})")

    prompt = f"""You are a strict document question-answering assistant.
Answer ONLY from the supplied context. 
If the answer is not available in the context, say exactly:
"I could not find this information in the uploaded documents." Do not invent facts.

Context:
{context_str}

Question: {query}
Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text, list(set(sources_str))
    except Exception as e:
        return f"Error communicating with AI model: {str(e)}", []

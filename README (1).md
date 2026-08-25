# Domain-Specific RAG Chatbot for PDF Question Answering

A Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, FAISS, Sentence Transformers, and Google Gemini.

## Features
- Upload multiple PDF documents.
- Automatic text extraction and chunking.
- Vector search powered by FAISS and Sentence Transformers (`all-MiniLM-L6-v2`).
- Grounded answers generated strictly from retrieved context.
- Fallback guardrails when information is absent.
- Displays source PDF file names and page numbers.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`

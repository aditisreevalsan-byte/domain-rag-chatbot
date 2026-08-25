import streamlit as st
import os
from rag_pipeline import process_pdfs, retrieve_context, generate_answer

st.set_page_config(page_title="Domain RAG Chatbot", layout="wide")

st.title("📚 Domain-Specific RAG Chatbot")
st.write("Upload course notes, manuals, or policies to ask grounded questions!")

# Sidebar for Settings and File Upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Extracting text and building vector store..."):
            index, chunks, metadata = process_pdfs(uploaded_files)
            if index is not None:
                st.session_state["vector_index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["metadata"] = metadata
                st.success(f"Processed {len(uploaded_files)} PDF(s) successfully!")
            else:
                st.error("Could not extract readable text from the uploaded PDFs.")

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display Chat History
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption("📌 **Sources:** " + ", ".join(message["sources"]))

# Chat Input
user_query = st.chat_input("Ask a question about your uploaded PDFs...")

if user_query:
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if "vector_index" not in st.session_state:
        answer = "Please upload PDF documents and click **Process Documents** first."
        sources = []
    else:
        with st.spinner("Searching documents & generating answer..."):
            retrieved_chunks, retrieved_meta = retrieve_context(
                user_query, 
                st.session_state["vector_index"], 
                st.session_state["chunks"], 
                st.session_state["metadata"]
            )
            answer, sources = generate_answer(
                user_query, 
                retrieved_chunks, 
                retrieved_meta, 
                api_key
            )

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption("📌 **Sources:** " + ", ".join(sources))

import streamlit as st
import pickle
import os
import gdown
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# File setup
FAISS_PATH = "tech10_faiss_index.pkl"
FILE_ID = "16bn_7iXySmif_Skn15gdnuZ07HVUp1xm"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download FAISS index if it doesn't exist
if not os.path.exists(FAISS_PATH):
    st.info("Downloading FAISS index from Google Drive...")
    gdown.download(URL, FAISS_PATH, quiet=False)

# Load FAISS index
with open(FAISS_PATH, "rb") as f:
    db = pickle.load(f)

retriever = db.as_retriever()
llm = pipeline("text-generation", model="google/flan-t5-small", tokenizer="google/flan-t5-small")

st.title("Finance LLM SEC Q&A")
query = st.text_input("Ask a financial question about Apple, Microsoft, etc.")

if query:
    docs = retriever.invoke(query)
    context = docs[0].page_content if docs else ""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt, max_length=100, do_sample=False)[0]["generated_text"]
    st.write(response.split("Answer:")[-1].strip())

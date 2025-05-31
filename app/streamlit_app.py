
import streamlit as st
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Load FAISS index
with open("faiss_index/tech10_faiss_index.pkl", "rb") as f:
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

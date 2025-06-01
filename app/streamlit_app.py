# streamlit_app.py
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import io
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import gdown
import os

st.set_page_config(page_title="📊 Finance RAG Chat & Charts", layout="wide")
st.title("💬 Ask a Financial Question | 📈 See the Chart")

# Download FAISS index if not exists
FAISS_PATH = "faiss_index/tech10_faiss_index_cpu.pkl"
gdown_id = "1ckak8qZYSKKUZu9Fq_Trp7qo692WOmJu"
if not os.path.exists(FAISS_PATH):
    st.info("Downloading FAISS index from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={gdown_id}", FAISS_PATH, quiet=False)

# Load FAISS index
def load_faiss():
    with open(FAISS_PATH, "rb") as f:
        return pickle.load(f)

db = load_faiss()
retriever = db.as_retriever()

# Load Hugging Face model (Phi-2 or other lightweight model)
@st.cache_resource
def load_model():
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# User prompt
user_input = st.text_area("Ask a question about tech company finances, or request a chart")

# Extract and execute code block
def extract_and_run_code(text):
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    for code in code_blocks:
        try:
            with st.echo():
                exec(code, globals())
            return True
        except Exception as e:
            st.error(f"Error running generated code: {e}")
            return False
    return False

if st.button("Submit") and user_input:
    with st.spinner("Processing with RAG + LLM..."):
        result = qa_chain(user_input)
        response = result["result"]

    # Show LLM answer
    st.markdown("### 🧠 Answer")
    st.code(response)

    # Attempt to run and display chart
    st.markdown("### 📈 Visualization Output")
    success = extract_and_run_code(response)
    if not success:
        st.info("No valid chart code detected or chart execution failed.")

    # Optional: Show source docs
    with st.expander("📄 Source Documents"):
        for doc in result["source_documents"]:
            st.markdown(doc.page_content[:500] + "...")

st.sidebar.markdown("Built with FAISS, LangChain, Hugging Face Transformers & Streamlit")

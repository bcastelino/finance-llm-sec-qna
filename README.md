# Finance LLM SEC Q&A

This project builds a question-answering system using 10-K filings from top tech companies (like Apple, Amazon, Microsoft, etc.). It leverages SEC-API, sentence-transformers, FAISS, Hugging Face LLMs, and Streamlit for a full-stack GenAI app.

## 🔧 Components

- **Data Extraction**: Pulls sections 1, 1A, 6, 7, 8 from latest 10-K filings
- **Embedding + Vector DB**: Uses `all-MiniLM-L6-v2` + FAISS
- **RAG Pipeline**: Combines FAISS retriever with `microsoft/phi-2` LLM using LangChain
- **Chart Generation**: Auto-detects matplotlib code in answers and renders charts
- **Streamlit App**: User-friendly interface to ask questions and see visual + textual answers
- **Lightweight + Free**: Runs on Colab, Streamlit Cloud, or low-spec local machines

## 🗂️ Folder Structure

```
finance-llm-sec-qna/
├── notebooks/               # Colab notebooks for data extraction and CPU conversion
├── faiss_index/             # (Optional) Folder to hold downloaded FAISS index
├── app/                     # Streamlit app with RAG + chart rendering
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Quickstart

1. Run `notebooks/SEC_Tech10_Extractor.ipynb` in Google Colab to build FAISS index
2. Upload your FAISS index to Google Drive (or use the one shared below)
3. Launch the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   The app will auto-download the FAISS index if not found locally using the provided public Drive link.

## 📦 FAISS Index Link

[Download tech10_faiss_index_cpu.pkl](https://drive.google.com/file/d/1ckak8qZYSKKUZu9Fq_Trp7qo692WOmJu/view?usp=sharing)

Make sure to update the file ID in the `streamlit_app.py` if you replace the index.

## 🧠 Example Questions

- "What were the top risks mentioned by Apple?"
- "Plot Microsoft’s net income from 2020 to 2023"
- "How did Meta explain its R&D investments in the MD&A section?"

## 🔓 Model

- Using `microsoft/phi-2` (lightweight, open-source) from Hugging Face
- Can be swapped for Mistral-7B or Gemma on Colab Pro or RunPod

---
Built with LangChain · FAISS · Hugging Face · Streamlit

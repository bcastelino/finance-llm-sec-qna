# Finance LLM SEC Q&A

This project builds a question-answering system using 10-K filings from top tech companies (like Apple, Amazon, Microsoft, etc.). It leverages SEC-API, sentence-transformers, FAISS, and Streamlit for a full-stack AI app.

## 🔧 Components

- **Data Extraction**: Pulls sections 1, 1A, 6, 7, 8 from latest 10-K filings.
- **Embedding + Vector DB**: Uses `all-MiniLM-L6-v2` and FAISS.
- **Q&A UI**: Streamlit interface for asking financial questions.
- **Cloud-first**: Built in Google Colab + hosted via Replit.

## 🗂️ Folder Structure

```
finance-llm-sec-qna/
├── notebooks/               # Colab notebook to extract & embed
├── faiss_index/             # Precomputed FAISS index
├── app/                     # Streamlit UI for Q&A
├── requirements.txt         # Dependencies
└── README.md
```

## 🚀 Quickstart

1. Run `notebooks/SEC_Tech10_Extractor.ipynb` in Google Colab
2. Save the FAISS index to `faiss_index/`
3. Run `app/streamlit_app.py` in Replit or locally using:
```bash
streamlit run app/streamlit_app.py
```

## 🧠 Example Questions

- "What were the top risks mentioned by Apple?"
- "How did NVIDIA describe their R&D expense?"
- "What are IBM’s cash flow highlights?"

---

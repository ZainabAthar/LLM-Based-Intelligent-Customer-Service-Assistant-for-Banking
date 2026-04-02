# NUST Bank Intelligent Customer Service Assistant

An LLM-powered banking assistant built with **Qwen-1.5-4B-Chat**, fine-tuned with LoRA on NUST Bank product data. Features a RAG (Retrieval-Augmented Generation) pipeline with ChromaDB, cross-encoder reranking, and a real-time streaming interface.

---

## Project Structure

```
llm-project-final/
├── app/
│   ├── api_service.py          # FastAPI backend (runs on Colab)
│   └── streamlit_app.py        # Streamlit frontend (runs locally)
├── preprocessed_data/
│   ├── bank_data_master.jsonl   # Processed Q&A pairs
│   └── rate_sheet_only.jsonl    # Rate sheet data
├── lora_bank_model_qwen/        # Fine-tuned LoRA adapter weights
├── vector_store_new/            # ChromaDB vector database (auto-generated)
├── extract_rate_sheet_jsonl.py  # Extracts rate sheets from Excel
├── preprocessing_new.py         # Converts Excel to structured JSONL
├── vector_db_setup.py           # Builds ChromaDB from JSONL
├── finetune.py                  # LoRA fine-tuning script
├── rag_assistant.py             # Core RAG engine + LLM generation
├── guardrails.py                # Input safety filter
├── product_info.xlsx            # Raw bank product data (Excel)
├── bank_qa_categories.json      # Mobile app Q&A categories
└── llm_proj.ipynb               # Colab notebook (all-in-one)
```

---

## Prerequisites

- **Google Colab** with T4 GPU (free tier works)
- **Python 3.10+**
- **ngrok account** (free) for tunneling: https://ngrok.com

### Python Dependencies (Colab)

```bash
pip install transformers peft trl bitsandbytes datasets accelerate
pip install fastapi uvicorn pyngrok
pip install chromadb sentence-transformers
```

### Python Dependencies (Local)

```bash
pip install streamlit requests
```

---

## Setup Guide

### Step 1: Data Preprocessing (One-Time)

Run on Colab or locally. Converts the raw Excel data into structured JSONL:

```bash
python preprocessing_new.py
```

This creates `preprocessed_data/bank_data_master.jsonl`.

### Step 2: Build Vector Database (One-Time)

Indexes the JSONL data into ChromaDB for semantic search:

```bash
python vector_db_setup.py
```

This creates the `vector_store_new/` directory with the `bank_knowledge_base` collection.

### Step 3: Fine-Tune the Model (One-Time, on Colab)

Upload the data files to Colab and run the fine-tuning script:

```bash
python finetune.py
```

This creates the LoRA adapter in `lora_bank_model_qwen/`. After training, back it up to Google Drive.

### Step 4: Start the API Backend (Colab)

```bash
python app/api_service.py --token {YOUR_NGROK_AUTH_TOKEN}
```

You will see:
```
YOUR ENDPOINT IS ALIVE!
Tunnel URL: https://xxx.ngrok-free.app
```

Copy the tunnel URL.

### Step 5: Start the Frontend (Local)

```bash
cd app
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser, paste the tunnel URL from Step 4, and start chatting.

---

## Architecture

```
┌─────────────────┐       HTTPS/ngrok       ┌─────────────────────┐
│   Streamlit UI  │ ◄──────────────────────► │  FastAPI + Qwen-4B  │
│   (Local PC)    │                          │  (Google Colab T4)  │
└─────────────────┘                          └─────────┬───────────┘
                                                       │
                                             ┌─────────▼───────────┐
                                             │  ChromaDB + Reranker│
                                             │  (Vector Search)    │
                                             └─────────────────────┘
```

1. User enters a query in the Streamlit chat interface
2. Query is sent to the FastAPI backend via ngrok tunnel
3. Guardrails filter validates the query
4. RAG pipeline retrieves top-10 relevant documents from ChromaDB
5. Cross-encoder reranks results for precision
6. Qwen-4B (with LoRA adapter) generates a streamed response
7. Tokens are streamed back to the frontend in real-time

---

## Key Features

- **LoRA Fine-Tuning**: Qwen-1.5-4B-Chat fine-tuned on NUST Bank product data
- **RAG Pipeline**: ChromaDB vector search + cross-encoder reranking
- **Real-Time Streaming**: Token-by-token response delivery
- **Product Glossary**: 30+ banking product acronyms built into the system prompt
- **Smart Guardrails**: Blocks off-topic queries while allowing natural conversation
- **Stateless Architecture**: Frontend manages chat history for reliable multi-turn conversations

---

## Configuration

Key settings in `rag_assistant.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DB_DIR` | `./vector_store_new` | ChromaDB database path |
| `LORA_PATH` | `/content/drive/MyDrive/colab_deploy/lora_bank_model_qwen` | LoRA adapter path |
| `BASE_MODEL` | `Qwen/Qwen1.5-4B-Chat` | Base language model |
| `max_new_tokens` | `1024` | Maximum response length |
| `temperature` | `0.7` | Response creativity |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Collection does not exist` | Run `python vector_db_setup.py` first |
| `No GPU found` | Switch Colab runtime to T4 GPU |
| `Connection failed` in Streamlit | Check ngrok tunnel URL, restart API |
| Short/unformatted responses | Ensure the latest `rag_assistant.py` is uploaded to Colab |

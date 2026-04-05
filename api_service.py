"""
api_service.py — NUST Bank AI — FastAPI streaming endpoint
"""
import uvicorn
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from test_rag import BankRAGAssistant
from document_parser import parse_file_to_records

app = FastAPI(title="NUST Bank AI", version="2.0")

assistant: Optional[BankRAGAssistant] = None


# ─────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    prompt:     str
    history:    List[Message] = Field(default_factory=list)
    session_id: str = "default"


class UploadRequest(BaseModel):
    file_bytes: str  # Base64 encoded file content
    filename:   str
    account_name: str = "Uploaded Document"
    session_id: str = "default"


class ClearRequest(BaseModel):
    session_id: str = "default"


# ─────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global assistant
    print("Starting NUST Bank AI API...")
    assistant = BankRAGAssistant()
    print("API ready.")


@app.get("/")
def root():
    return {"status": "running", "model": "Qwen2.5-3B-Instruct + LoRA"}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": assistant is not None}


# ─────────────────────────────────────────────
#  Chat endpoint (streaming)
# ─────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    if not assistant:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        # history = [{"role": m.role, "content": m.content} for m in req.history]

        # Pass prompt and session_id as a dict to the assistant.chat()
        query_data = {"prompt": req.prompt, "session_id": req.session_id}
        history = [{"role": m.role, "content": m.content} for m in req.history]

        generator, top_docs = assistant.chat(query_data, history)

        sources = [
            f"{i + 1}. {meta.get('account_name', 'NUST Bank Record')}"
            for i, (_, _, meta) in enumerate(top_docs)
        ]

        # Deduplicate sources (same account can appear multiple times)
        seen = set()
        unique_sources = []
        for s in sources:
            label = s.split(". ", 1)[-1]
            if label not in seen:
                seen.add(label)
                unique_sources.append(s)

        def stream():
            # Stream answer tokens
            for token in generator:
                yield token

            # Append sources footer after the answer
            if unique_sources:
                yield "\n\n---\n**Sources:**\n"
                for s in unique_sources:
                    yield f"- {s}\n"

        return StreamingResponse(stream(), media_type="text/plain")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  Upload / Document Ingestion endpoint
# ─────────────────────────────────────────────
@app.post("/upload")
async def upload(req: UploadRequest):
    if not assistant:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        import base64
        # Decode the base64 file content
        file_bytes = base64.b64decode(req.file_bytes)
        
        # Parse document using document_parser
        records = parse_file_to_records(file_bytes, req.filename, req.account_name)
        
        if not records:
            return {"status": "error", "message": "No readable text found in document."}

        # Ingest into RAG assistant
        print(f"Ingesting {req.filename} into RAG...")
        count = assistant.ingest_records(records, session_id=req.session_id)
        print("Ingestion successful.")
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {count} segments from {req.filename}",
            "chunks": count
        }
    except Exception as e:
        print(f"UPLOAD ERROR: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/clear_uploads")
async def clear_uploads(req: ClearRequest):
    if not assistant:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    success = assistant.clear_session_records(req.session_id)
    return {"status": "success" if success else "error"}


# ─────────────────────────────────────────────
#  Optional: ngrok tunnel (for Colab / remote)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from pyngrok import ngrok

        NGROK_TOKEN = "3BwmKW3xK0fxUwinusx45GIUNpM_2BNWEYm7hpB3TvHbi8Uwf"  # replace with your token
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(8000).public_url
        print("\n" + "=" * 50)
        print("PUBLIC ENDPOINT:")
        print(public_url)
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"ngrok not available: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
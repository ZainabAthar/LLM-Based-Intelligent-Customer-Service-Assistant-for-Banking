"""
api_service.py — NUST Bank AI — FastAPI streaming endpoint
"""
import uvicorn
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_assistant import BankRAGAssistant

app = FastAPI(title="NUST Bank AI", version="2.0")

assistant: Optional[BankRAGAssistant] = None


# ─────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    prompt:  str
    history: List[Message] = Field(default_factory=list)


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
        # Convert Pydantic Message objects to plain dicts for the assistant
        history = [{"role": m.role, "content": m.content} for m in req.history]

        # FIX: capture docs BEFORE entering the stream closure.
        # retrieve_context is called inside assistant.chat() and returns top_docs.
        # We extract sources here so they're available when stream() runs.
        generator, top_docs = assistant.chat(req.prompt, history)

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
#  Optional: ngrok tunnel (for Colab / remote)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from pyngrok import ngrok

        NGROK_TOKEN = "3Bnzldsllzo22ZLaMrgUFo5xRdB_41DMsF8Ab3cZJgtKJZMjX"  # replace with your token
        ngrok.set_auth_token(NGROK_TOKEN)
        public_url = ngrok.connect(8000).public_url
        print("\n" + "=" * 50)
        print("PUBLIC ENDPOINT:")
        print(public_url)
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"ngrok not available: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000)